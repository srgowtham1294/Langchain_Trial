from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")

information = """
Thomas Cruise Mapother IV (born July 3, 1962) is an American actor and film producer. Regarded as a Hollywood icon,[1][2][3] he has received various accolades, including an Honorary Palme d'Or, an Academy Honorary Award, and three Golden Globe Awards, in addition to nominations for four competitive Academy Awards.[4][5] As of 2025, his films have grossed more than $13.3 billion worldwide,[6] placing him among the highest-grossing actors of all time.[7] One of Hollywood's most bankable stars, he is consistently one of the world's highest-paid actors.[8]

Cruise began acting in the early 1980s and made his breakthrough with leading roles in Risky Business (1983) and Top Gun (1986), the latter earning him a reputation as a sex symbol.[9] Critical acclaim came with his roles in the dramas The Color of Money (1986), Rain Man (1988), and Born on the Fourth of July (1989). For his portrayal of Ron Kovic in the latter, he won a Golden Globe Award and received a nomination for the Academy Award for Best Actor. As a leading Hollywood star in the 1990s, he starred in commercially successful films, including the drama A Few Good Men (1992), the thriller The Firm (1993), the horror film Interview with the Vampire (1994), and the sports comedy-drama Jerry Maguire (1996); for the latter, he won a Golden Globe Award for Best Actor and his second nomination for the Academy Award for Best Actor. Cruise's performance in the drama Magnolia (1999) earned him another Golden Globe Award and a nomination for the Academy Award for Best Supporting Actor.

Cruise subsequently established himself as a star of science fiction and action films, often performing his own risky stunts. He played fictional agent Ethan Hunt in the Mission: Impossible film series.[10] His other films in the genre include Vanilla Sky (2001), Minority Report (2002), The Last Samurai (2003), Collateral (2004), War of the Worlds (2005), Knight and Day (2010), Jack Reacher (2012), Oblivion (2013), Edge of Tomorrow (2014), and Top Gun: Maverick (2022).

Cruise holds the Guinness World Record for the most consecutive $100-million-grossing movies, a feat achieved with eleven films released between 2012 and 2025.[11] In December 2024, he was awarded the U.S. Navy's highest civilian honor, the Distinguished Public Service Award, in recognition of his "outstanding contributions" to the military, with his screen roles.[12] In March 2025, he was named the recipient of the British Film Institute Fellowship, the BFI's highest honor, for his contributions to cinema.[13] Forbes ranked him as the world's most powerful celebrity in 2006.[14] He was named People's Sexiest Man Alive in 1990,[15] and received the top honor of "Most Beautiful People" in 1997.[16] Outside his film career, Cruise has been an outspoken advocate for the Church of Scientology, which has resulted in controversy and scrutiny of his involvement in the organization. An aviation enthusiast, he has held a pilot certificate since 1994.[17]
"""

summary_template = """
given the information {information} I've given about a person I want you to create:
1. A short summary
2. 2 random facts about them in bullet points.

"""

summary_prompt_template = PromptTemplate(
    input_variables=['information'],template=summary_template
)

#llm = ChatGroq(temperature=0.4, model='llama-3.3-70b-versatile')
llm_goog = ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=google_api_key, temperature=0.5)

chain = summary_prompt_template | llm_goog

response = chain.invoke(input={'information':information})

print(response.content)

