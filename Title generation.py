!pip install transformers
from transformers import BartTokenizer, BartForConditionalGeneration

def generate_title_from_paragraph(paragraph):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + paragraph, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return title

if __name__ == "__main__":
    paragraph = """
    Armaan Malik (born 22 July 1995) is an Indian singer, songwriter, record producer, voice-over, performer & actor. He is known for his singing in multiple languages, including Hindi, Telugu, English, Bengali, Kannada, Marathi, Tamil, Gujarati, Punjabi, Urdu and Malayalam. In 2006, he took part in Sa Re Ga Ma Pa L'il Champs but was eliminated after finishing in 8th position . He is the brother of music composer Amaal Mallik. Previously represented by Universal Music India and T-Series, he is now represented by Arista Records.[1][2][3] He has his own record label called Always Music Global in partnership with Warner Music India.[3] His first on-screen appearance was in the film Kaccha Limboo in 2011.
    Malik started singing at the age of 4. He competed on Sa Re Ga Ma Pa L'il Champs in 2006, eventually finishing 8th.[5] Later he learnt Indian classical music for 10 years. Malik made his debut as a child singer in Bollywood in 2007 with "Bum Bum Bole" in Taare Zameen Par, under the musical direction of the Shankar-Ehsaan-Loy.[6][7]

Malik has dubbed for My Name Is Khan for the English boy and also lent voice for the character Salim in the radio version of Slumdog Millionaire for BBC Radio 1. In 2014, he made his debut as a playback singer singing "Tumko Toh Aana Hi Tha" in the Hindi-language movie Jai Ho. The movie featured two more songs, "Love You Till The End (House Mix)" and the title track, "Jai Ho" also sung by him.[8][9] Apart from singing, Malik and his music composer brother Amaal Mallik also featured in the beginning of Jai Ho in the song "Love You Till The End". In the same year, he sang "Naina" with Sona Mohapatra for the film Khoobsurat and "Auliya" for Ungli.[citation needed]

In 2015, he sang "Main Hoon Hero Tera" for Hero, "Kwahishein" for Calendar Girls and "Tumhe Aapna Banane Ka" for Hate Story 3 which his brother Amaal Malik composed. The latter one he sang with Neeti Mohan. He also sang another song for Hate Story 3 titled "Wajah Tum Ho" under Baman's composition. He sang "Yaar Indha Muyalkutti" by D Imaan. He also sang a single "Main Rahoon Ya Na Rahoon" under Amaal's composition. He was awarded Filmfare R. D. Burman Award for New Music Talent in that year.[citation needed]

In 2016, Malik sang for the films Mastizaade, Sanam Re, Kapoor & Sons, Azhar, Do Lafzon Ki Kahani and "Sab Tera" with Shraddha Kapoor for Baaghi under Amaal's composition. He sang "Foolishq" with Shreya Ghoshal for Ki & Ka, he worked with Jeet Gannguli for the song "Mujhko Barsaat Bana Lo" for Junooniyat and also sang his first Bengali song "Dhitang Dhitang" for Love Express under Jeet's composition. He was the lead singer of the film M.S. Dhoni: The Untold Story. He sang four songs for Hindi soundtrack and three songs for Tamil soundtrack of that film under Amaal's composition. He sang "Sau Asmaan" with Neeti Mohan for Baar Baar Dekho and "Ishaara" for Force 2 under Amaal's composition. He sang "Tum Jo Mille" for Saansein, "Pal Pal Dil Ke Paas Reprise" and "Dil Mein Chupa Lunga Remake" for Wajah Tum Ho. The latter one was composed by Meet Bros and the last two songs, he sang with Tulsi Kumar. He sang a single "Pyaar Manga Hain Remake" with Neeti Mohan. He also sang the title track of Star Paarivar Awards 2016 with Palak Muchhal and Meet Bros under Meet Bros composition.[10]

In 2019, he lent his voice for two songs including "Jab Se Mera Dil" with Palak Muchhal for the movie Amavas, "Dil Me Ho Tum" for the movie Why Cheat India, "Kyun Rabba" for the movie Badla. Malik also got featured as a coach on the Indian version of the reality show The Voice becoming the youngest Indian singer to be a coach on the show. He sang "Chale Aana" in De De Pyaar De composed by Amaal and written by Kunaal Verma. The song was well received by the audience in general. Malik voiced the titular character in the Hindi-dubbed version of Disney's Aladdin, a live action remake of the 1992 movie, Aladdin.
    """
  generated_title = generate_title_from_paragraph(paragraph)
  print("Generated Title:", generated_title)
