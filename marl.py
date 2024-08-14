import asyncio
import pandas as pd
from common import Paper, Prompt, Conference, ChatAgent


class MARLPaper(Paper):
    @property
    def prompt(self) -> Prompt:
        return Prompt(
            sys_msg={
                "role": "system",
                "content": "You will play the role of a researcher in Multi Agent Reinforcement Learning."
                           "-----"
                           "###TASK###:"
                           "You will be provided with a paper title related to field of Multi Agent"
                           " Reincorement Learning. Go through the paper and assign it a topic (no more than 2 words), "
                           "highlight the problem it is solving, solution paper has proposed, benchmarks used, "
                           "challenges in multi agent RL paper is addressing, future work to extend this paper."
            },
            usr_msg={
                "role": "user",
                "content": f"Paper title is {self.title}"
                           "Generate the response in form of dictionary which can be parsed with ast"
                           "with the following structure,"
                           '{"Title": <Tile of the paper>, "Topic": <Topic of the paper, don\'t copy title>",\
                           "Problem":"", "Solution": "", "Benchmarks": "", "Challenges": ""}',
            }
        )


if __name__ == "__main__":
    chat_agent = ChatAgent()
    confs = [
        Conference("Neurips_2023", "https://neurips.cc/virtual/2023/papers.html?filter=titles"),
        Conference("ICML_2024", "https://icml.cc/virtual/2024/papers.html?filter=titles")
    ]

    for conf in confs:
        conf.get_titles("multi-agent")
        conf.papers = [MARLPaper(x) for x in conf.titles]

    for conf in confs:
        conf.summaries = asyncio.run(chat_agent.process_papers(conf.papers))
        pd.DataFrame(conf.summaries).to_excel(str(conf) + '.xlsx', index=False)
