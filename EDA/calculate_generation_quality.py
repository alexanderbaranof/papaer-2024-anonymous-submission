from datetime import datetime
from collections import Counter
import re
from urllib.parse import unquote

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

# Configs
print("Script last run:", datetime.now())
print()
print()


def clear_title(title):

    if "ru.wiktionary" in title:
        title = title.split("https://ru.wiktionary.org/wiki/")[1]
    elif "wikipedia" in title:
        title = title.split("https://ru.wikipedia.org/wiki/")[1]
    elif "en.wiktionary" in title:
        title = title.split("https://en.wiktionary.org/wiki/")[1]
    
    title = title.replace("_", " ")
    title = title.lower()
    title = title.replace("ё", "е")
    title = re.sub(r'\(.*?\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    return title


def clear_explain(explain):
    explain = explain.lower()
    explain = explain.replace("ё", "е")
    explain = re.sub(r'[^\w\s]', '', explain)
    return explain

# Read data
df = pd.read_csv("/home/ambaranov/paper-2024/Data/Dataset.csv")
df_explain = pd.read_excel("/home/ambaranov/paper-2024/Data/all_explain_quality.xlsx")

df_explain = df_explain[["title", "gigachat_pred_explain", "yandexgpt_pred_explain", "Mistral_explain", "gpt4o_explain", "gpt4o_explain_big"]]

df_explain["gigachat_pred_explain"] = df_explain["gigachat_pred_explain"].apply(clear_explain)
df_explain["yandexgpt_pred_explain"] = df_explain["yandexgpt_pred_explain"].apply(clear_explain)
df_explain["Mistral_explain"] = df_explain["Mistral_explain"].apply(clear_explain)
df_explain["gpt4o_explain"] = df_explain["gpt4o_explain"].apply(clear_explain)
df_explain["gpt4o_explain_big"] = df_explain["gpt4o_explain_big"].apply(clear_explain)

df = df.merge(df_explain, on="title")
df["is_url"] = df["URL"].apply(lambda x: True if "wikipedia.org" in str(x) or "wiktionary.org" in str(x) else False)
df = df[df["is_url"]]

print("examples with url", df.shape)

df["URL"] = df["URL"].apply(unquote)
df["URL_title"] = df["URL"].apply(clear_title)

df["gigachat_pred_explain_simple_score"] = df.apply(lambda row: True if row["URL_title"] in row["gigachat_pred_explain"] else False, axis=1)
df["yandexgpt_pred_explain_simple_score"] = df.apply(lambda row: True if row["URL_title"] in row["yandexgpt_pred_explain"] else False, axis=1)
df["Mistral_explain_simple_score"] = df.apply(lambda row: True if row["URL_title"] in row["Mistral_explain"] else False, axis=1)
df["gpt4o_explain_simple_score"] = df.apply(lambda row: True if row["URL_title"] in row["gpt4o_explain"] else False, axis=1)
df["gpt4o_explain_big_simple_score"] = df.apply(lambda row: True if row["URL_title"] in row["gpt4o_explain_big"] else False, axis=1)


print("GigaChat", round(df["gigachat_pred_explain_simple_score"].mean(), 2), df["gigachat_pred_explain_simple_score"].sum())
print("YaGPT", round(df["yandexgpt_pred_explain_simple_score"].mean(), 2), df["yandexgpt_pred_explain_simple_score"].sum())
print("Mistral", round(df["Mistral_explain_simple_score"].mean(), 2), df["Mistral_explain_simple_score"].sum())
print("Gpt4o", round(df["gpt4o_explain_simple_score"].mean(), 2), df["gpt4o_explain_simple_score"].sum())
print("gpt4o-big", round(df["gpt4o_explain_big_simple_score"].mean(), 2), df["gpt4o_explain_big_simple_score"].sum())