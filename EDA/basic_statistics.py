from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

# Configs
print("Script last run:", datetime.now())
print()
print()

# Functions
def clear_title(title):
    slash_position = title.find("//")
    return title[:slash_position].strip()

def get_serious_part(title):
    slash_position = title.find("//")
    return title[slash_position+2:].strip()

# Read data
df = pd.read_csv("/home/ambaranov/paper-2024/Data/Dataset.csv")

# Checkers

assert np.sum(df["title"].isna()) == 0 # Checking for missing data values
assert np.sum(df["summary"].isna()) == 0 # Checking for missing data values
assert np.sum(df["is_word_play"].isna()) == 0 # Checking for missing data values
assert set(df["is_word_play"].unique().tolist()) == set(["Да", "Нет"]) # Checking for correctness of the filled in field 
assert len(df["title"].unique().tolist()) == df.shape[0] # Checking that all headings are unique



# Word and symbol statistics
df["title_before"] = df["title"].apply(clear_title)
df["title_after"] = df["title"].apply(get_serious_part)

df["title_symbol_len"] = df["title"].apply(lambda x: len(x))
df["title_before_symbol_len"] = df["title_before"].apply(lambda x: len(x))
df["title_after_symbol_len"] = df["title_after"].apply(lambda x: len(x))

df["title_tokens"] = df["title"].apply(lambda x: word_tokenize(x))
df["title_before_tokens"] = df["title_before"].apply(lambda x: word_tokenize(x))
df["title_after_tokens"] = df["title_after"].apply(lambda x: word_tokenize(x))

df["title_tokens_len"] = df["title_tokens"].apply(lambda x: len(x))
df["title_before_tokens_len"] = df["title_before_tokens"].apply(lambda x: len(x))
df["title_after_tokens_len"] = df["title_after_tokens"].apply(lambda x: len(x))

# Split data
df_pos = df[df["is_word_play"] == "Да"]
df_neg = df[df["is_word_play"] == "Нет"]

# Save basic counts

all_samples = len(df)
all_positive_samples = len(df_pos)
all_negative_samples = len(df_neg)

assert all_samples == all_positive_samples + all_negative_samples

# all dataset
mean_symbol_len = df["title_symbol_len"].mean()
mean_symbol_len_before = df["title_before_symbol_len"].mean()
mean_symbol_len_after = df["title_after_symbol_len"].mean()

mean_token_len = df["title_tokens_len"].mean()
mean_token_len_before = df["title_before_tokens_len"].mean()
mean_token_len_after = df["title_after_tokens_len"].mean()

# positive set
mean_symbol_len_pos = df_pos["title_symbol_len"].mean()
mean_symbol_len_before_pos = df_pos["title_before_symbol_len"].mean()
mean_symbol_len_after_pos = df_pos["title_after_symbol_len"].mean()

mean_token_len_pos = df_pos["title_tokens_len"].mean()
mean_token_len_before_pos = df_pos["title_before_tokens_len"].mean()
mean_token_len_after_pos = df_pos["title_after_tokens_len"].mean()

# negative set
mean_symbol_len_neg = df_neg["title_symbol_len"].mean()
mean_symbol_len_before_neg = df_neg["title_before_symbol_len"].mean()
mean_symbol_len_after_neg = df_neg["title_after_symbol_len"].mean()

mean_token_len_neg = df_neg["title_tokens_len"].mean()
mean_token_len_before_neg = df_neg["title_before_tokens_len"].mean()
mean_token_len_after_neg = df_neg["title_after_tokens_len"].mean()

# First print
print("Размер датасета", all_samples)
print("Количество позитивных примеров", all_positive_samples)
print("Количество негативных примеров", all_negative_samples)

print("----------------------------------------------------------")

print("Статистика по длине Title во всем датасете")
print("Средняя длина title в символах", mean_symbol_len)
print("Средняя длина title до слеша в символах", mean_symbol_len_before)
print("Средняя длина title после слеша в символах", mean_symbol_len_after)
print("---")
print("Средняя длина title в словах", mean_token_len)
print("Средняя длина title до слеша в словах", mean_token_len_before)
print("Средняя длина title после слеша в словах", mean_token_len_after)

print("----------------------------------------------------------")

print("Статистика по длине Title в положительном таргете")
print("Средняя длина title в символах", mean_symbol_len_pos)
print("Средняя длина title до слеша в символах", mean_symbol_len_before_pos)
print("Средняя длина title после слеша в символах", mean_symbol_len_after_pos)
print("---")
print("Средняя длина title в словах", mean_token_len_pos)
print("Средняя длина title до слеша в словах", mean_token_len_before_pos)
print("Средняя длина title после слеша в словах", mean_token_len_after_pos)

print("----------------------------------------------------------")

print("Статистика по длине Title в негативном таргете")
print("Средняя длина title в символах", mean_symbol_len_neg)
print("Средняя длина title до слеша в символах", mean_symbol_len_before_neg)
print("Средняя длина title после слеша в символах", mean_symbol_len_after_neg)
print("---")
print("Средняя длина title в словах", mean_token_len_neg)
print("Средняя длина title до слеша в словах", mean_token_len_before_neg)
print("Средняя длина title после слеша в словах", mean_token_len_after_neg)
print("----------------------------------------------------------")

# Statistics using methods

all_methods = df_pos["Разметка - механизм 1"].tolist()
all_methods.extend(df_pos["Разметка - механизм 2"].dropna().tolist())

print("----------------------------------------------------------")
print("Статистика использования колчиества раз механизма:")
c = Counter(all_methods)
for k, v in sorted(c.items(), key=lambda i: i[1], reverse=True):
    print(k, v)
print("----------------------------------------------------------")


print("----------------------------------------------------------")
print("Статистика использования колчиества раз механизма с учетом сочетаний:")

first_method = df_pos["Разметка - механизм 1"].tolist()
second_method = df_pos["Разметка - механизм 2"].fillna("").tolist()

combine_methods = list()
for i in range(len(first_method)):
    combine_methods.append(
        "-".join(
            sorted([str(first_method[i]), str(second_method[i])], reverse=True)
        )
    )

c = Counter(combine_methods)
for k, v in sorted(c.items(), key=lambda i: i[1], reverse=True):
    print(k, v)
print("----------------------------------------------------------")