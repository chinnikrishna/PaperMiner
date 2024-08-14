from openai import AsyncOpenAI

import os
import ast
import asyncio
from asyncio.locks import Semaphore
import requests
from typing import Dict, List, Union
from dataclasses import dataclass
from bs4 import BeautifulSoup

from tqdm import tqdm
from config import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


@dataclass
class Prompt:
    sys_msg: Dict
    usr_msg: Dict


@dataclass
class Paper:
    title: str

    @property
    def prompt(self) -> Prompt:
        raise NotImplemented


class Conference:
    def __init__(self, conf_name: str, conf_url: str) -> None:
        self.conf_name: str = conf_name
        self.url: str = conf_url
        self.titles: List[str] = []
        self.papers: List[Paper] = []
        self.summaries: List[Dict] = []

    def __repr__(self) -> str:
        return self.conf_name

    def __len__(self) -> int:
        return len(self.titles)

    def get_titles(self, title_filter=""):
        """
        Override this method to change parsing of conference
        website
        """
        response = requests.get(self.url)
        response.raise_for_status()

        # Parse with Beautiful Soup
        parsed_page = BeautifulSoup(response.text, features='html.parser')

        # Get titles of papers and filter them
        for li in parsed_page.find_all('li'):
            a_tag = li.find('a', href=True)
            if a_tag and 'poster' in a_tag['href']:
                title = a_tag.get_text(strip=True)
                if title_filter != "":
                    if title_filter in title.lower():
                        self.titles.append(title)
                else:
                    self.titles.append(title)


class ChatAgent:
    def __init__(self, rem_requests: int = 5000, rem_tokens: int = 60000,
                 toks_thres: int = 6000, retries: int = 3) -> None:
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.rem_requests: int = rem_requests
        self.rem_tokens: int = rem_tokens
        self.toks_thresh: int = toks_thres
        self.sleep_time: float = 0.0
        self.sem: Semaphore = asyncio.Semaphore(50)
        self.gpt_model: str = 'gpt-4o'
        self.timeout: int = 120
        self.retries: int = retries

    async def api_call(self, prompt: Prompt) -> Union[Dict, Exception]:
        async with self.sem:
            # Check the limits before making an API call
            if self.rem_requests < 5 or self.rem_tokens < self.toks_thresh:
                print(f"Limit reached. Sleeping for {self.sleep_time}")
                await asyncio.sleep(self.sleep_time)
            try:
                messages = [prompt.sys_msg, prompt.usr_msg]
                resp = await asyncio.wait_for(
                    self.client.chat.completions.with_raw_response.create(
                        model=self.gpt_model, messages=messages), timeout=self .timeout)
                self.rem_requests = int(resp.headers.get('x-ratelimit-remaining-requests'))
                self.rem_tokens = int(resp.headers.get('x-ratelimit-remaining-tokens'))
                req_sleep_time = resp.headers.get('x-ratelimit-reset-requests')
                toks_sleep_time = resp.headers.get('x-ratelimit-reset-tokens')
                self.sleep_time = max(self.conv_time(req_sleep_time), self.conv_time(toks_sleep_time))
                return self.process_result(resp)
            except Exception as e:
                print(e)
                return e

    def process_result(self, api_response):
        """
        Override this method to process response in a different way
        """
        try:
            comp = api_response.parse()
            data = comp.choices[0].message.content
            data = ast.literal_eval(data[9:-3])
            return data
        except Exception as e:
            print(e)
            return e

    def conv_time(self, str_time):
        if 'ms' in str_time:
            return float(str_time.split('ms')[0])
        elif 's' in str_time:
            return float(str_time.split('s')[0])
        else:
            raise ValueError(f'Unknown time string {str_time}')

    async def _process_paper(self, paper: Paper):
        result = await self.api_call(paper.prompt)
        return paper, result

    async def _batch_process_papers(self, batch: List[Paper]):
        tasks = [self._process_paper(x) for x in batch]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            yield await task

    async def process_papers(self, papers: List[Paper]):
        to_process: List[Paper] = papers
        answers = []
        for _ in range(self.retries):
            re_process: List[Paper] = []
            async for paper, ans in self._batch_process_papers(to_process):
                if isinstance(ans, Exception):
                    re_process.append(paper)
                else:
                    answers.append(ans)
            to_process = re_process
            if not to_process:
                break
        if len(to_process) != 0:
            print("Failed to process the following papers")
            for paper in to_process:
                print(paper.title)
        return answers
