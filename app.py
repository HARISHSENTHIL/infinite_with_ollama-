import io
import os
import re
import time
from itertools import islice
from functools import partial
from multiprocessing.pool import ThreadPool
from queue import Queue, Empty
from typing import Callable, Iterable, Iterator, Optional, TypeVar
from ollama import Client


import gradio as gr
import pandas as pd
import requests.exceptions
from huggingface_hub import create_repo, whoami, DatasetCard


model_id ="llama3"
client = Client(host='')
save_dataset_hf_token = os.environ.get("SAVE_DATASET_HF_TOKEN")

MAX_TOTAL_NB_ITEMS = 100  # almost infinite, don't judge me (actually it's because gradio needs a fixed number of components)
MAX_NB_ITEMS_PER_GENERATION_CALL = 10
NUM_ROWS = 100
NUM_VARIANTS = 10
NAMESPACE = "mysyntheticdata"
URL = "https://huggingface.co/spaces/Harish-as-harry/mysyntheticdata1"

GENERATE_DATASET_NAMES_FOR_SEARCH_QUERY = (
        "A Machine Learning Practioner is looking for a dataset that matches '{search_query}'. "
        f"Generate a list of {MAX_NB_ITEMS_PER_GENERATION_CALL} names of quality dataset that don't exist but sound plausible and would "
        "be helpful. Feel free to reuse words from the query '{search_query}' to name the datasets. "
        "Every dataset should be about '{search_query}' and have descriptive tags/keywords including the ML task name associated to the dataset (classification, regression, anomaly detection, etc.). Use the following format:\n1. DatasetName1 (tag1, tag2, tag3)\n1. DatasetName2 (tag1, tag2, tag3)"
)

GENERATE_DATASET_CONTENT_FOR_SEARCH_QUERY_AND_NAME_AND_TAGS = (
    "A ML practitioner is looking for a dataset CSV after the query '{search_query}'. "
    "Generate the first 5 rows of a plausible and quality CSV for the dataset '{dataset_name}'. "
    "You can get inspiration from related keywords '{tags}' but most importantly the dataset should correspond to the query '{search_query}'. "
    "Focus on quality text content and and use a 'label' or 'labels' column if it makes sense (invent labels, avoid reusing the keywords, be accurate while labelling texts). "
    "Reply using a short description of the dataset with title **Dataset Description:** followed by the CSV content in a code block and with title **CSV Content Preview:**."
)
GENERATE_MORE_ROWS = "Can you give me 10 additional samples in CSV format as well ? Use the same CSV header '{csv_header}'."
GENERATE_VARIANTS_WITH_RARITY_AND_LABEL = "Focus on generating samples for the label '{label}' and ideally generate {rarity} samples."
GENERATE_VARIANTS_WITH_RARITY = "Focus on generating {rarity} samples."

RARITIES = ["pretty obvious", "common/regular", "unexpected but useful", "uncommon but still plausible", "rare/niche but still plausible"]
LONG_RARITIES = [
    "obvious",
    "expected",
    "common",
    "regular",
    "unexpected but useful"
    "original but useful",
    "specific but not far-fetched",
    "uncommon but still plausible",
    "rare but still plausible",
    "very nice but still plausible",
]

landing_page_datasets_generated_text = """
1. NewsEventsPredict (classification, media, trend)
2. FinancialForecast (economy, stocks, regression)
3. HealthMonitor (science, real-time, anomaly detection)
4. SportsAnalysis (classification, performance, player tracking)
5. SciLiteracyTools (language modeling, science literacy, text classification)
6. RetailSalesAnalyzer (consumer behavior, sales trend, segmentation)
7. SocialSentimentEcho (social media, emotion analysis, clustering)
8. NewsEventTracker (classification, public awareness, topical clustering)
9. HealthVitalSigns (anomaly detection, biometrics, prediction)
10. GameStockPredict (classification, finance, sports contingency)
"""
default_output = landing_page_datasets_generated_text.strip().split("\n")
assert len(default_output) == MAX_NB_ITEMS_PER_GENERATION_CALL

DATASET_CARD_CONTENT = """
---
license: mit
tags:
- infinite-dataset-hub
- synthetic
---

{title}

_Note: This is an AI-generated dataset so its content may be inaccurate or false_

{content}

**Source of the data:**

The dataset was generated using the [Infinite Dataset Hub]({url}) and {model_id} using the query '{search_query}':

- **Dataset Generation Page**: {dataset_url}
- **Model**: https://huggingface.co/{model_id}
- **More Datasets**: https://huggingface.co/datasets?other=infinite-dataset-hub
"""

css = """
a {
    color: var(--body-text-color);
}

.datasetButton {
    justify-content: start;
    justify-content: left;
}
.tags {
    font-size: var(--button-small-text-size);
    color: var(--body-text-color-subdued);
}
.topButton {
    justify-content: start;
    justify-content: left;
    text-align: left;
    background: transparent;
    box-shadow: none;
    padding-bottom: 0;
}
.topButton::before {
    content: url("data:image/svg+xml,%3Csvg style='color: rgb(209 213 219)' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' aria-hidden='true' focusable='false' role='img' width='1em' height='1em' preserveAspectRatio='xMidYMid meet' viewBox='0 0 25 25'%3E%3Cellipse cx='12.5' cy='5' fill='currentColor' fill-opacity='0.25' rx='7.5' ry='2'%3E%3C/ellipse%3E%3Cpath d='M12.5 15C16.6421 15 20 14.1046 20 13V20C20 21.1046 16.6421 22 12.5 22C8.35786 22 5 21.1046 5 20V13C5 14.1046 8.35786 15 12.5 15Z' fill='currentColor' opacity='0.5'%3E%3C/path%3E%3Cpath d='M12.5 7C16.6421 7 20 6.10457 20 5V11.5C20 12.6046 16.6421 13.5 12.5 13.5C8.35786 13.5 5 12.6046 5 11.5V5C5 6.10457 8.35786 7 12.5 7Z' fill='currentColor' opacity='0.5'%3E%3C/path%3E%3Cpath d='M5.23628 12C5.08204 12.1598 5 12.8273 5 13C5 14.1046 8.35786 15 12.5 15C16.6421 15 20 14.1046 20 13C20 12.8273 19.918 12.1598 19.7637 12C18.9311 12.8626 15.9947 13.5 12.5 13.5C9.0053 13.5 6.06886 12.8626 5.23628 12Z' fill='currentColor'%3E%3C/path%3E%3C/svg%3E");
    margin-right: .25rem;
    margin-left: -.125rem;
    margin-top: .25rem;
}
.bottomButton {
    justify-content: start;
    justify-content: left;
    text-align: left;
    background: transparent;
    box-shadow: none;
    font-size: var(--button-small-text-size);
    color: var(--body-text-color-subdued);
    padding-top: 0;
    align-items: baseline;
}
.bottomButton::before {
    content: 'tags:';
    margin-right: .25rem;
}
.buttonsGroup {
    background: transparent;
}
.buttonsGroup:hover {
    background: var(--input-background-fill);
}
.buttonsGroup div {
    background: transparent;
}
.insivibleButtonGroup {
    display: none;
}

@keyframes placeHolderShimmer{
    0%{
        background-position: -468px 0
    }
    100%{
        background-position: 468px 0
    }
}
.linear-background {
    animation-duration: 1s;
    animation-fill-mode: forwards;
    animation-iteration-count: infinite;
    animation-name: placeHolderShimmer;
    animation-timing-function: linear;
    background-image: linear-gradient(to right, var(--body-text-color-subdued) 8%, #dddddd11 18%, var(--body-text-color-subdued) 33%);
    background-size: 1000px 104px;
    color: transparent;
    background-clip: text;
}
.settings {
    background: transparent;
}
.settings button span {
    color: var(--body-text-color-subdued);
}
"""


with gr.Blocks(css=css) as demo:
    generated_texts_state = gr.State((landing_page_datasets_generated_text,))
    with gr.Column() as search_page:
        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown(
                    "# 🤗 Infinite Dataset Hub ♾️\n\n"
                    "An endless catalog of datasets, created just for you.\n\n"
                )
                with gr.Row():
                    search_bar = gr.Textbox(max_lines=1, placeholder="Search datasets, get infinite results", show_label=False, container=False, scale=9)
                    search_button = gr.Button("🔍", variant="primary", scale=1)
                button_groups: list[gr.Group] = []
                buttons: list[gr.Button] = []
                for i in range(MAX_TOTAL_NB_ITEMS):
                    if i < len(default_output):
                        line = default_output[i]
                        dataset_name, tags = line.split(".", 1)[1].strip(" )").split(" (", 1)
                        group_classes = "buttonsGroup"
                        dataset_name_classes = "topButton"
                        tags_classes = "bottomButton"
                    else:
                        dataset_name, tags = "⬜⬜⬜⬜⬜⬜", "░░░░, ░░░░, ░░░░"
                        group_classes = "buttonsGroup insivibleButtonGroup"
                        dataset_name_classes = "topButton linear-background"
                        tags_classes = "bottomButton linear-background"
                    with gr.Group(elem_classes=group_classes) as button_group:
                        button_groups.append(button_group)
                        buttons.append(gr.Button(dataset_name, elem_classes=dataset_name_classes))
                        buttons.append(gr.Button(tags, elem_classes=tags_classes))

                load_more_datasets = gr.Button("Load more datasets")  # TODO: dosable when reaching end of page
                gr.Markdown(f"_powered by [{model_id}](https://huggingface.co/{model_id})_")
            with gr.Column(scale=4, min_width="200px"):
                with gr.Accordion("Settings", open=False, elem_classes="settings"):
                    gr.Markdown("Save datasets to your account")
                    gr.LoginButton()
                    select_namespace_dropdown = gr.Dropdown(choices=[NAMESPACE], value=NAMESPACE, label="Select user or organization", visible=False)
                    gr.Markdown("Save datasets as public or private datasets")
                    visibility_radio = gr.Radio(["public", "private"], value="public", container=False, interactive=False)
    with gr.Column(visible=False) as dataset_page:
        gr.Markdown(
            "# 🤗 Infinite Dataset Hub ♾️\n\n"
            "An endless catalog of datasets, created just for you.\n\n"
        )
        dataset_title = gr.Markdown()
        gr.Markdown("_Note: This is an AI-generated dataset so its content may be inaccurate or false_")
        dataset_content = gr.Markdown()
        generate_full_dataset_button = gr.Button("Generate Full Dataset", variant="primary")
        dataset_dataframe = gr.DataFrame(visible=False, interactive=False, wrap=True)
        save_dataset_button = gr.Button("💾 Save Dataset", variant="primary", visible=False)
        open_dataset_message = gr.Markdown("", visible=False)
        dataset_share_button = gr.Button("Share Dataset URL")
        dataset_share_textbox = gr.Textbox(visible=False, show_copy_button=True, label="Copy this URL:", interactive=False, show_label=True)
        back_button = gr.Button("< Back", size="sm")

    ###################################
    #
    #       Utils
    #
    ###################################

    T = TypeVar("T")

    def batched(it: Iterable[T], n: int) -> Iterator[list[T]]:
        it = iter(it)
        while batch := list(islice(it, n)):
            yield batch


    def stream_reponse(msg: str, generated_texts: tuple[str] = (), max_tokens=500) -> Iterator[str]:
        messages = [
            {"role": "user", "content": msg}
        ] + [
            item
            for generated_text in generated_texts
            for item in [
                {"role": "assistant", "content": generated_text},
                {"role": "user", "content": "Can you generate more ?"},
            ]
        ]
        for _ in range(3):
            try:
                for message in client.chat(
                    messages=messages,
                    model=model_id,
                    stream=True,
                    
                ):
                    if 'message' in message:
                        msg_content = message['message'].get('content', '')
                        if msg_content:
                            yield msg_content
                        else:
                            print("Error: 'content' missing in 'message'")
                    else:
                        print("Error: 'message' key missing in response")
    
            except requests.exceptions.ConnectionError as e:
                print(e + "\n\nRetrying in 1sec")
                time.sleep(1)
                continue
            break


    def gen_datasets_line_by_line(search_query: str, generated_texts: tuple[str] = ()) -> Iterator[str]:
        search_query = search_query or ""
        search_query = search_query[:1000] if search_query.strip() else ""
        generated_text = ""
        current_line = ""
        for token in stream_reponse(
            GENERATE_DATASET_NAMES_FOR_SEARCH_QUERY.format(search_query=search_query),
            generated_texts=generated_texts,
        ):
            current_line += token
            if current_line.endswith("\n"):
                yield current_line
                generated_text += current_line
                current_line = ""
        yield current_line
        generated_text += current_line
        print("-----\n\n" + generated_text)


    def gen_dataset_content(search_query: str, dataset_name: str, tags: str) -> Iterator[str]:
        search_query = search_query or ""
        search_query = search_query[:1000] if search_query.strip() else ""
        generated_text = ""
        for token in stream_reponse(GENERATE_DATASET_CONTENT_FOR_SEARCH_QUERY_AND_NAME_AND_TAGS.format(
            search_query=search_query,
            dataset_name=dataset_name,
            tags=tags,
        ), max_tokens=1500):
            generated_text += token
            yield generated_text
        print("-----\n\n" + generated_text)


    def _write_generator_to_queue(queue: Queue, func: Callable[..., Iterable], kwargs: dict) -> None:
        for i, result in enumerate(func(**kwargs)):
            queue.put(result)
        return None


    def iflatmap_unordered(
        func: Callable[..., Iterable[T]],
        *,
        kwargs_iterable: Iterable[dict],
    ) -> Iterable[T]:
        queue = Queue()
        with ThreadPool() as pool:
            async_results = [
                pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_iterable
            ]
            try:
                while True:
                    try:
                        yield queue.get(timeout=0.05)
                    except Empty:
                        if all(async_result.ready() for async_result in async_results) and queue.empty():
                            break
            finally:
                # we get the result in case there's an error to raise
                [async_result.get(timeout=0.05) for async_result in async_results]
    
    def generate_partial_dataset(title: str, content: str, search_query: str, variant: str, csv_header: str, output: list[dict[str, str]], indices_to_generate: list[int], max_tokens=1500) -> Iterator[int]:
        dataset_name, tags = title.strip("# ").split("\ntags:", 1)
        dataset_name, tags = dataset_name.strip(), tags.strip()
        messages = [
            {
                "role": "user",
                "content": GENERATE_DATASET_CONTENT_FOR_SEARCH_QUERY_AND_NAME_AND_TAGS.format(
                    dataset_name=dataset_name,
                    tags=tags,
                    search_query=search_query,
                    )
                    },
                    {"role": "assistant", "content": title + "\n\n" + content},
                    {"role": "user", "content": GENERATE_MORE_ROWS.format(csv_header=csv_header) + " " + variant},
                    ]
        
        for _ in range(3):
            generated_text = ""
            generated_csv = ""
            current_line = ""
            nb_samples = 0
            _in_csv = False
            try:
                for message in client.chat(
                    model=model_id,  # Ensure this is set to the appropriate Ollama model
                    messages=messages,
                    stream=True,
                    ):
                    if nb_samples >= len(indices_to_generate):
                        break
                    
                    if 'message' in message:
                        msg_content = message['message'].get('content', '')
                        if msg_content:
                            current_line += msg_content
                            generated_text += msg_content
                            if current_line.endswith("\n"):
                                _in_csv = _in_csv ^ current_line.lstrip().startswith("```")
                                if current_line.strip() and _in_csv and not current_line.lstrip().startswith("```"):
                                    generated_csv += current_line
                                    try:
                                        generated_df = parse_csv_df(generated_csv.strip(), csv_header=csv_header)
                                        if len(generated_df) > nb_samples:
                                            output[indices_to_generate[nb_samples]] = generated_df.iloc[-1].to_dict()
                                            nb_samples += 1
                                            yield 1
                                    except Exception:
                                        pass
                                current_line = ""
                    else:
                        print("Error: 'content' missing in 'message'")
                else:
                    print("Error: 'message' key missing in response")

            except requests.exceptions.ConnectionError as e:
                print(e + "\n\nRetrying in 1sec")
                time.sleep(1)
                continue
                break


        # for debugging
        # with open(f".output{indices_to_generate[0]}.txt", "w") as f:
        #     f.write(generated_text)


    def generate_variants(preview_df: pd.DataFrame):
        label_candidate_columns = [column for column in preview_df.columns if "label" in column.lower()]
        if label_candidate_columns:
            labels = preview_df[label_candidate_columns[0]].unique()
            if len(labels) > 1:
                return [
                    GENERATE_VARIANTS_WITH_RARITY_AND_LABEL.format(rarity=rarity, label=label)
                    for rarity in RARITIES
                    for label in labels
                ]
        return [
            GENERATE_VARIANTS_WITH_RARITY.format(rarity=rarity)
            for rarity in LONG_RARITIES
        ]


    def parse_preview_df(content: str) -> tuple[str, pd.DataFrame]:
        _in_csv = False
        csv = "\n".join(
            line for line in content.split("\n") if line.strip()
            and (_in_csv := (_in_csv ^ line.lstrip().startswith("```")))
            and not line.lstrip().startswith("```")
        )
        if not csv:
            raise gr.Error("Failed to parse CSV Preview")
        return csv.split("\n")[0], parse_csv_df(csv)


    def parse_csv_df(csv: str, csv_header: Optional[str] = None) -> pd.DataFrame:
        # Fix generation mistake when providing a list that is not in quotes
        for match in re.finditer(r'''(?!")\[(["'][\w ]+["'][, ]*)+\](?!")''', csv):
            span = match.string[match.start() : match.end()]
            csv = csv.replace(span, '"' + span.replace('"', "'") + '"', 1)
        # Add header if missing
        if csv_header and csv.strip().split("\n")[0] != csv_header:
            csv = csv_header + "\n" + csv
        # Read CSV
        df = pd.read_csv(io.StringIO(csv), skipinitialspace=True)
        return df


    ###################################
    #
    #       Buttons
    #
    ###################################


    def _search_datasets(search_query):
        yield {generated_texts_state: []}
        yield {
            button_group: gr.Group(elem_classes="buttonsGroup insivibleButtonGroup")
            for button_group in button_groups[MAX_NB_ITEMS_PER_GENERATION_CALL:]
        }
        yield {
            k: v
            for dataset_name_button, tags_button in batched(buttons, 2)
            for k, v in {
                dataset_name_button: gr.Button("⬜⬜⬜⬜⬜⬜", elem_classes="topButton linear-background"),
                tags_button: gr.Button("░░░░, ░░░░, ░░░░", elem_classes="bottomButton linear-background")
            }.items()
        }
        current_item_idx = 0
        generated_text = ""
        for line in gen_datasets_line_by_line(search_query):
            if "I'm sorry" in line or "against Microsoft's use case policy" in line:
                raise gr.Error("Error: inappropriate content")
            if current_item_idx >= MAX_NB_ITEMS_PER_GENERATION_CALL:
                return
            if line.strip() and line.strip().split(".", 1)[0].isnumeric():
                try:
                    dataset_name, tags = line.strip().split(".", 1)[1].strip(" )").split(" (", 1)
                except ValueError:
                    dataset_name, tags = line.strip().split(".", 1)[1].strip(" )").split(" ", 1)
                dataset_name, tags = dataset_name.strip("()[]* "), tags.strip("()[]* ")
                generated_text += line
                yield {
                    buttons[2 * current_item_idx]: gr.Button(dataset_name, elem_classes="topButton"),
                    buttons[2 * current_item_idx + 1]: gr.Button(tags, elem_classes="bottomButton"),
                    generated_texts_state: (generated_text,),
                }
                current_item_idx += 1


    @search_button.click(inputs=search_bar, outputs=button_groups + buttons + [generated_texts_state])
    def search_dataset_from_search_button(search_query):
        yield from _search_datasets(search_query)


    @search_bar.submit(inputs=search_bar, outputs=button_groups + buttons + [generated_texts_state])
    def search_dataset_from_search_bar(search_query):
        yield from _search_datasets(search_query)


    @load_more_datasets.click(inputs=[search_bar, generated_texts_state], outputs=button_groups + buttons + [generated_texts_state])
    def search_more_datasets(search_query, generated_texts):
        current_item_idx = initial_item_idx = len(generated_texts) * MAX_NB_ITEMS_PER_GENERATION_CALL
        yield {
            button_group: gr.Group(elem_classes="buttonsGroup")
            for button_group in button_groups[len(generated_texts) * MAX_NB_ITEMS_PER_GENERATION_CALL:(len(generated_texts) + 1) * MAX_NB_ITEMS_PER_GENERATION_CALL]
        }
        generated_text = ""
        for line in gen_datasets_line_by_line(search_query, generated_texts=generated_texts):
            if "I'm sorry" in line or "against Microsoft's use case policy" in line:
                raise gr.Error("Error: inappropriate content")
            if current_item_idx - initial_item_idx >= MAX_NB_ITEMS_PER_GENERATION_CALL:
                return
            if line.strip() and line.strip().split(".", 1)[0].isnumeric():
                try:
                    dataset_name, tags = line.strip().split(".", 1)[1].strip(" )").split(" (", 1)
                except ValueError:
                    dataset_name, tags = line.strip().split(".", 1)[1].strip(" )").split(" ", 1) [0], ""
                dataset_name, tags = dataset_name.strip("()[]* "), tags.strip("()[]* ")
                generated_text += line
                yield {
                    buttons[2 * current_item_idx]: gr.Button(dataset_name, elem_classes="topButton"),
                    buttons[2 * current_item_idx + 1]: gr.Button(tags, elem_classes="bottomButton"),
                    generated_texts_state: (*generated_texts, generated_text),
                }
                current_item_idx += 1

    def _show_dataset(search_query, dataset_name, tags):
        yield {
            search_page: gr.Column(visible=False),
            dataset_page: gr.Column(visible=True),
            dataset_title: f"# {dataset_name}\n\n tags: {tags}",
            dataset_share_textbox: gr.Textbox(visible=False),
            dataset_dataframe: gr.DataFrame(visible=False),
            generate_full_dataset_button: gr.Button(interactive=True),
            save_dataset_button: gr.Button(visible=False),
            open_dataset_message: gr.Markdown(visible=False)
        }
        for generated_text in gen_dataset_content(search_query=search_query, dataset_name=dataset_name, tags=tags):
            yield {dataset_content: generated_text}


    show_dataset_inputs = [search_bar, *buttons]
    show_dataset_outputs = [search_page, dataset_page, dataset_title, dataset_content, generate_full_dataset_button, dataset_dataframe, save_dataset_button, open_dataset_message, dataset_share_textbox]
    scroll_to_top_js = """
    function (...args) {
        console.log(args);
        if ('parentIFrame' in window) {
            window.parentIFrame.scrollTo({top: 0, behavior:'smooth'});
        } else {
            window.scrollTo({ top: 0 });
        }
        return args;
    }
    """

    def show_dataset_from_button(search_query, *buttons_values, i):
        dataset_name, tags = buttons_values[2 * i : 2 * i + 2]
        yield from _show_dataset(search_query, dataset_name, tags)
    
    for i, (dataset_name_button, tags_button) in enumerate(batched(buttons, 2)):
        dataset_name_button.click(partial(show_dataset_from_button, i=i), inputs=show_dataset_inputs, outputs=show_dataset_outputs, js=scroll_to_top_js)
        tags_button.click(partial(show_dataset_from_button, i=i), inputs=show_dataset_inputs, outputs=show_dataset_outputs, js=scroll_to_top_js)


    @back_button.click(outputs=[search_page, dataset_page], js=scroll_to_top_js)
    def show_search_page():
        return gr.Column(visible=True), gr.Column(visible=False)


    @generate_full_dataset_button.click(inputs=[dataset_title, dataset_content, search_bar, select_namespace_dropdown, visibility_radio], outputs=[dataset_dataframe, generate_full_dataset_button, save_dataset_button])
    def generate_full_dataset(title, content, search_query, namespace, visability):
        dataset_name, tags = title.strip("# ").split("\ntags:", 1)
        dataset_name, tags = dataset_name.strip(), tags.strip()
        csv_header, preview_df = parse_preview_df(content)
        # Remove dummy "id" columns
        for column_name, values in preview_df.to_dict(orient="series").items():
            try:
                if [int(v) for v in values] == list(range(len(preview_df))):
                    preview_df = preview_df.drop(columns=column_name)
                if [int(v) for v in values] == list(range(1, len(preview_df) + 1)):
                    preview_df = preview_df.drop(columns=column_name)
            except Exception:
                pass
        columns = list(preview_df)
        output: list[Optional[dict]] = [None] * NUM_ROWS
        output[:len(preview_df)] = [{"idx": i, **x} for i, x in enumerate(preview_df.to_dict(orient="records"))]
        yield {
            dataset_dataframe: gr.DataFrame(pd.DataFrame([{"idx": i, **x} for i, x in enumerate(output) if x]), visible=True),
            generate_full_dataset_button: gr.Button(interactive=False),
            save_dataset_button: gr.Button(f"💾 Save Dataset {namespace}/{dataset_name}" + (" (private)" if visability != "public" else ""), visible=True, interactive=False)
        }
        kwargs_iterable = [
            {
                "title": title,
                "content": content,
                "search_query": search_query,
                "variant": variant,
                "csv_header": csv_header,
                "output": output,
                "indices_to_generate": list(range(len(preview_df) + i, NUM_ROWS, NUM_VARIANTS)),
            }
            for i, variant in enumerate(islice(generate_variants(preview_df), NUM_VARIANTS))
        ]
        for _ in iflatmap_unordered(generate_partial_dataset, kwargs_iterable=kwargs_iterable):
            yield {dataset_dataframe: pd.DataFrame([{"idx": i, **{column_name: x.get(column_name) for column_name in columns}} for i, x in enumerate(output) if x])}
        yield {save_dataset_button: gr.Button(interactive=True)}
        print(f"Generated {dataset_name}!")


    @save_dataset_button.click(inputs=[dataset_title, dataset_content, search_bar, dataset_dataframe, select_namespace_dropdown, visibility_radio], outputs=[save_dataset_button, open_dataset_message])
    def save_dataset(title: str, content: str, search_query: str, df: pd.DataFrame, namespace: str, visability: str, oauth_token: Optional[gr.OAuthToken]):
        dataset_name, tags = title.strip("# ").split("\ntags:", 1)
        dataset_name, tags = dataset_name.strip(), tags.strip()
        token = oauth_token.token if oauth_token else save_dataset_hf_token
        repo_id = f"{namespace}/{dataset_name}"
        dataset_url = f"{URL}?q={search_query.replace(' ', '+')}&dataset={dataset_name.replace(' ', '+')}&tags={tags.replace(' ', '+')}"
        gr.Info("Saving dataset...")
        yield {save_dataset_button: gr.Button(interactive=False)}
        create_repo(repo_id=repo_id, repo_type="dataset", private=visability!="public", exist_ok=True, token=token)
        df.to_csv(f"hf://datasets/{repo_id}/data.csv", storage_options={"token": token}, index=False)
        DatasetCard(DATASET_CARD_CONTENT.format(title=title, content=content, url=URL, dataset_url=dataset_url, model_id=model_id, search_query=search_query)).push_to_hub(repo_id=repo_id, repo_type="dataset", token=token)
        gr.Info(f"✅ Dataset saved at {repo_id}")
        additional_message = "PS: You can also save datasets under your account in the Settings ;)"
        yield {open_dataset_message: gr.Markdown(f"# 🎉 Yay ! Your dataset has been saved to [{repo_id}](https://huggingface.co/datasets/{repo_id}) !\n\nDataset link: [https://huggingface.co/datasets/{repo_id}](https://huggingface.co/datasets/{repo_id})\n\n{additional_message}", visible=True)}
        print(f"Saved {dataset_name}!")


    @dataset_share_button.click(inputs=[dataset_title, search_bar], outputs=[dataset_share_textbox])
    def show_dataset_url(title, search_query):
        dataset_name, tags = title.strip("# ").split("\ntags:", 1)
        dataset_name, tags = dataset_name.strip(), tags.strip()
        return gr.Textbox(
            f"{URL}?q={search_query.replace(' ', '+')}&dataset={dataset_name.replace(' ', '+')}&tags={tags.replace(' ', '+')}",
            visible=True,
        )

    @demo.load(outputs=show_dataset_outputs + button_groups + buttons + [generated_texts_state] + [select_namespace_dropdown, visibility_radio])
    def load_app(request: gr.Request, oauth_token: Optional[gr.OAuthToken]):
        if oauth_token:
            user_info = whoami(oauth_token.token)
            yield {
                select_namespace_dropdown: gr.Dropdown(
                        choices=[user_info["name"]] + [org_info["name"] for org_info in user_info["orgs"]],
                        value=user_info["name"],
                        visible=True,
                    ),
                visibility_radio: gr.Radio(interactive=True),
                }
        query_params = dict(request.query_params)
        if "dataset" in query_params:
            yield from _show_dataset(
                search_query=query_params.get("q", query_params["dataset"]),
                dataset_name=query_params["dataset"],
                tags=query_params.get("tags", "")
            )
        elif "q" in query_params:
            yield {search_bar: query_params["q"]}
            yield from _search_datasets(query_params["q"])
        else:
            yield {search_page: gr.Column(visible=True)}


demo.launch()