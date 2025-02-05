import argparse
import os
import os.path as osp
import sys
import subprocess
import shutil
import requests
import re
import json
import fitz
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from subprocess import TimeoutExpired
from openai import OpenAI, RateLimitError, APIError

MAX_ITERS = 10

def generate_latex_files_with_citation(base_dir, arxiv_id):
    def get_bibtex_from_arxiv(arxiv_id):
        """
        Given an arXiv URL, fetch the BibTeX citation.
        """
        
        bibtex_url = f'https://arxiv.org/bibtex/{arxiv_id}'
        
        response = requests.get(bibtex_url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error fetching BibTeX citation (HTTP status code {response.status_code}).")

    def insert_bibtex_citation(tex_content, new_citation):
        start_index = tex_content.find('\\begin{filecontents}{references.bib}')
        
        if start_index == -1:
            raise ValueError("Could not find references.bib section in the LaTeX file")
        
        end_index = tex_content.find('\\end{filecontents}', start_index)
        
        updated_content = (
            tex_content[:end_index] + 
            '\n' + new_citation + 
            '\n' + 
            tex_content[end_index:]
        )
        
        return updated_content

    latex_source = osp.join("templates", "nanoGPT", "latex")
    new_latex = osp.join(base_dir, "latex")
    shutil.copytree(latex_source, new_latex)
    print(f"Copied '{latex_source}' to '{new_latex}'.")
    
    with open(osp.join(new_latex, "template.tex"), 'r') as file:
        tex_content = file.read()

    new_citation = get_bibtex_from_arxiv(arxiv_id)
    updated_tex_content = insert_bibtex_citation(tex_content, new_citation)

    with open(osp.join(new_latex, "template.tex"), 'w') as file:
        file.write(updated_tex_content)


def download_pdf(arxiv_url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pdf_url = arxiv_url.replace("abs", "pdf")
    arxiv_id = arxiv_url.split("/")[-1]
    filename = arxiv_id + ".pdf"
    pdf_path = os.path.join(save_dir, filename)
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return arxiv_id, filename, pdf_path
    else:
        raise Exception(f"Failed to download PDF from {arxiv_url}")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def find_github_url(text):
    """
    Finds GitHub URLs in text, including:
    - https://github.com/...
    - http://www.github.com/...
    - www.github.com/... (without protocol)
    - github.com/... (without protocol or www)
    """
    github_url_pattern = r"(https?://)?(www\.)?github\.com/[^\s]+"
    match = re.search(github_url_pattern, text)
    return match.group(0) if match else None

def construct_prompt(pdf_filename, github_url):
    prompt = f"You are an ambitious AI/ML PhD student highly proficient in Python coding. Read the attached paper ({pdf_filename}). Now, complete the following tasks."
    prompt += "\nTask 1: Pick a single simple experiment from the paper to reproduce in Python. Describe your experiment plan/pseudocode in detail, starting your response with \"This is an experiment\"... Always surround this experiment description in ```description and ``` tags."

    prompt += "\n\nTask 2: Come up with a 1-3 word short title for the experiment. Always surround this title in ```title and ``` tags. This will be used to name a folder, so no whitespaces allowed."

    prompt += "\n\nTask 3: implement the experiment. Your response should be pure Python code. If needed, use PyTorch and not TensorFlow. Surround your response to this task inside ```experiment and ``` tags."
    if github_url:
        prompt += " To help you, I have provided source code (a Git repo) in a special formatted version in repomix-output.txt. Understand this file, and simply reproduce the Python code corresponding to the experiment you chose. Although you may have to combine code from different files, everything you need is in repomix-output.txt."
    prompt += "\n\nHere are some specific instructions for your code which you must follow:"
    prompt += "\nCommand line instructions: in your main function, you should parse any relevant command line arguments to run your code. MAKE SURE THEY HAVE DEFAULT VALUES.  You will need to import argparse. One argument you must always parse is the output directory. In your main function, use the following code:\n"
    prompt += "parser = argparse.ArgumentParser()\nparser.add_argument(\"--out_dir\", type=str, default=\"run_0\")"
    prompt += "\n\nExperiment results formatting instructions: it will be important to track performance metrics such as training loss, test accuracy, etc. You should keep your results in a JSON, with following structure:\n"
    prompt += "final_info = {\"<experiment_or_dataset_name>\":{\"means\": {\"<metric_name_1>_mean\": value1, \"<metric_name_2>_mean\": value2}}}. Save this using the following code:\n"
    prompt += "with open(os.path.join(args.out_dir, \"final_info.json\"), \"w\") as f:\n    json.dump(final_infos, f)"
    prompt += "\n\nAlso keep a json called all_results. The difference is that while final_info just stores the means (so each metric is mapped to a single value), all_results stores the full history of the experiment (so each metric is mapped to a list of values).\n"
    prompt += "all_results = {\"<experiment_or_dataset_name>\":\"<metric_name_1>\":[value1, value2, ..., valueN], \"<metric_name_2>\":[value1, value2, ..., valueN]}}. Save this using the following code:\n"
    prompt += "with open(os.path.join(args.out_dir, \"all_results.npy\"), \"wb\") as f:\n    np.save(f, all_results)"
    prompt += "\n\nYour experiment results will be automatically parsed, so you must copy the above code EXACTLY. Don't change names, don't leave anything out."
    prompt += "\nFinal instructions for this task: DO NOT USE DUMMY PLACEHOLDERS, as we will be directly running your code as a baseline experiment for a research paper. Implement everything required for the experiment."

    prompt += "\n\nTask 4: write code (which will be run in a separate file, so add necessarry import statements at the beginning) to load the above result files (final_info.json and all_results.npy) and plot the results. Surround your response to this task inside ```plotting and ``` tags." 
    prompt += "\nThe following code will be helpful:\n"
    prompt += "for folder in folders:\n" \
              "    if folder.startswith(\"run\") and osp.isdir(folder):\n" \
              "        results_dict = np.load(osp.join(folder, \"all_results.npy\"), allow_pickle=True).item()"
    prompt += "\n\nImportant: you should also incorporate the following code:\n"
    prompt += "labels = {\"run_0\": \"Baseline\",}"
    prompt += "\n\nThe key of this dictionary will be of the format \"run_{number}\", which will also be the name of the output directory of the experiment results (from Task 3)."
    prompt += "\nOnly the runs in the \"labels\" dictionary should be plotted; for each run, the label of the plot should be labels[run]. So your code should contain something like:\n"
    prompt += "plt.plot(iteration_number, metric_result, label=labels[run], color=random_color)"
    prompt += "\nFinal instructions for this task: DO NOT USE DUMMY PLACEHOLDERS, as we will be directly running your code to plot the results. Implement everything required to plot your experiment's results. For each metric you plot, save the figure in an appropriately named file (e.g. training_loss_{experiment_name}.png)."

    prompt += "\n\nTask 5: Finally, imagine that now, you are going to provide your code as a starting \"template\" for another student to build upon. Come up with a short (just a few sentences) description of your experiment code file from task 3, and also suggest some research directions for the student to explore (keep in mind that the student will use your code as a starting point). Keep your response to this task short. Surround your response to this part in ```instructions and ``` tags."

    prompt += "\n\nSo to summarize your instructions: you must complete 5 tasks. Task 1: experiment plan, Task 2: experiment title, Task 3: experiment code, Task 4: experiment results plotting, Task 5: instructions for the student."
    return prompt

def generate_and_save_template(client, assistant, arxiv_id, prompt, attachments=[]):
    def get_run_result(run):
        """Check for content filtering and other finish reasons"""
        if run.status == 'failed':
            if run.last_error and 'content_filter' in run.last_error.code.lower():
                return "Response blocked by content filter"
            return f"Run failed: {run.last_error.message}"
        return None

    try:
        thread = client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": prompt,
                "attachments": attachments
            }]
        )
        
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            timeout=30
        )

        if (status_message := get_run_result(run)):
            print(status_message)
            exit(1)
        
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc"
        )
        
        if not messages.data:
            print("No messages received")
            exit(1)
        
        message_content = messages.data[-1].content[0].text.value
        
        annotations = messages.data[-1].content[0].text.annotations
        for ann in annotations:
            if ann.type == 'file_citation' and 'content_filter' in ann.text.lower():
                print("Content filtered due to policy violations")
        
        print("Final response:", message_content)

        title_match = re.search(r'```title(.*?)```', message_content, re.DOTALL)
        description_match = re.search(r'```description(.*?)```', message_content, re.DOTALL)
        experiment_match = re.search(r'```experiment(.*?)```', message_content, re.DOTALL)
        plotting_match = re.search(r'```plotting(.*?)```', message_content, re.DOTALL)
        instructions_match = re.search(r'```instructions(.*?)```', message_content, re.DOTALL)
        
        if title_match and description_match and experiment_match and plotting_match and instructions_match:
            title = title_match.group(1).strip()
            base_dir = osp.join("templates", f"{title}_auto")
            os.makedirs(base_dir, exist_ok=False)
            print(f"Template directory successfully created")

            description_filename="experiment_description.txt"
            description = description_match.group(1).strip()
            with open(osp.join(base_dir, description_filename), 'w', encoding='utf-8') as f:
                f.write(description)
            print(f"Description successfully saved to {description_filename}")

            experiment_code_filename="experiment.py"
            experiment_code = experiment_match.group(1).strip()
            with open(osp.join(base_dir, experiment_code_filename), 'w') as f:
                f.write(experiment_code)
            print(f"Code successfully saved to {experiment_code_filename}")

            plotting_code_filename = "plot.py"
            plotting_code = plotting_match.group(1).strip()
            with open(osp.join(base_dir, plotting_code_filename), 'w') as f:
                f.write(plotting_code)
            print(f"Code successfully saved to {plotting_code_filename}")

            instructions_filename = "prompt.json"
            instructions = instructions_match.group(1).strip()
            data = {
                "system": "You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.",
                "task_description": instructions
            }
            with open(osp.join(base_dir, instructions_filename), "w") as f:
                json.dump(data, f, indent=4)
            print(f"Instructions successfully saved to {instructions_filename}")

            seed_ideas_filename = "seed_ideas.json"
            with open(osp.join(base_dir, seed_ideas_filename), "w") as f:
                f.write("[]")
            print(f"Empty JSON file {seed_ideas_filename} created.")

            generate_latex_files_with_citation(base_dir, arxiv_id)
            print("Latex files successfully generated")

            return base_dir
        
        else:
            with open("paper_prompt_response.txt", 'w', encoding='utf-8') as f:
                f.write(message_content)
            print("Some task block(s) not found - full response saved")
            exit(1)

    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except APIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def execute_experiment(folder_name, initial_run=True, timeout=7200):
    cwd = osp.abspath(folder_name)
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"run_0.py"),
    )
    command = [
        "python",
        "experiment.py",
        f"--out_dir=run_0",
    ]
    
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run 0 failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_0")):
                shutil.rmtree(osp.join(cwd, f"run_0"))
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            next_prompt = ""
            if initial_run:
                next_prompt = "This Python script re-implements an experiment from a research paper, described as follows:\nBEGIN DESCRIPTION\n"
                with open(osp.join(folder_name, "experiment_description.txt"), "r") as f:
                    next_prompt += f.read()
                next_prompt += "\nEND DESCRIPTION\n\n"
                next_prompt += f"When I tried to execute this code, I got the following error message: {stderr_output}"
                next_prompt += "\n\nPlease help me fix this error message by making any necessary changes to experiment.py."
            else:
                next_prompt = f"Run failed with the following error: {stderr_output}"
        else:
            print("Program finished successfully")
            next_prompt = ""
        return result.returncode, next_prompt
    
    except TimeoutExpired:
        print(f"Run 0 timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_0")):
            shutil.rmtree(osp.join(cwd, f"run_0"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt

def run_and_debug_code(base_dir):
    '''Aider automated debugging loop'''
    fnames = [osp.join(base_dir, "experiment.py")]
    main_model = Model("gpt-4-turbo-preview")
    coder = Coder.create(
        main_model=main_model,
        io=InputOutput(yes=True),
        fnames=fnames,
        stream=False,
        use_git=False,
        edit_format="diff",
    )

    current_iter = 0
    while True:
        return_code, next_prompt = execute_experiment(folder_name=base_dir, initial_run=(current_iter == 0))
        print(f"Prompt:\n{next_prompt}")
        current_iter += 1
        if return_code == 0 or current_iter >= MAX_ITERS:
            break
        _ = coder.run(next_prompt)
    if current_iter >= MAX_ITERS:
        print("Not able to debug program.")
        return False
    print("Program debugged successfully.")
    return True

def cleanup(scratch_space_dir, client):
    """Clean up all OpenAI assistants, files, and vector stores"""
    try:
        if client:
            assistants = client.beta.assistants.list(limit=100)
            for assistant in assistants.data:
                print(f"Deleting assistant: {assistant.id}")
                client.beta.assistants.delete(assistant.id)

            files = client.files.list()
            for file in files.data:
                print(f"Deleting file: {file.id}")
                client.files.delete(file.id)

            vector_stores = client.beta.vector_stores.list(limit=100)
            for vs in vector_stores.data:
                print(f"Deleting vector store: {vs.id}")
                client.beta.vector_stores.delete(vs.id)
        
        if scratch_space_dir:
            if os.path.exists(scratch_space_dir):
                try:
                    shutil.rmtree(scratch_space_dir)
                except Exception as e:
                    print(f"An error occurred while deleting the directory: {e}")
            else:
                print(f"The directory {scratch_space_dir} does not exist.")

        return "Cleanup completed successfully"
    
    except Exception as e:
        return f"Cleanup error: {str(e)}"


if __name__ == "__main__":
    client = None
    scratch_space_dir = None
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument("--exp_dir", type=str, help="Name of the experiment directory (something like {idea}_auto)", default=None)
    args = parser.parse_args()

    try:
        if args.exp_dir:
            base_dir = os.path.join('templates', args.exp_dir)
            if not os.path.isdir(base_dir):
                raise ValueError(f"Experiment directory not found: {base_dir}")
        else:
            scratch_space_dir = 'assistant_scratch_space'
            os.makedirs(scratch_space_dir, exist_ok=True)
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            assistant = client.beta.assistants.create(
                name="Research Assistant",
                instructions="You are a PhD student studying AI and ML. Follow the instructions from the user.",
                model="gpt-4-turbo",
                tools=[{"type": "file_search"},{"type": "code_interpreter"}],
            )

            arxiv_url = input("Enter arxiv paper URL: ")
            arxiv_id, filename, pdf_path = download_pdf(arxiv_url, scratch_space_dir)

            paper_file = client.files.create(file=open(pdf_path, "rb"), purpose="assistants")
            attachments = [{"file_id": paper_file.id, "tools": [{"type": "file_search"}]}]

            pdf_text = extract_text_from_pdf(pdf_path)
            github_url = find_github_url(pdf_text)
            if github_url:
                print(f"Found GitHub URL: {github_url}")
                command = ["repomix", "--remote", github_url]
                subprocess.run(command, cwd=scratch_space_dir, stderr=subprocess.PIPE, check=True, shell=True)
                code_file = client.files.create(file=open(os.path.join(scratch_space_dir, "repomix-output.txt"), "rb"), purpose="assistants")
                attachments.append({"file_id": code_file.id, "tools": [{"type": "file_search"}]})
            
            base_dir = generate_and_save_template(client, assistant, arxiv_id=arxiv_id, prompt=construct_prompt(filename, github_url), attachments=attachments)
            
        run_and_debug_code(base_dir)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
    finally:
        print(cleanup(scratch_space_dir, client))