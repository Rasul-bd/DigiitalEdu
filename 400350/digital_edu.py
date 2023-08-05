#создай здесь свой индивидуальный проект!
#создай здесь свой индивидуальный
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

df["city"].fillna("-1", inplace = True)

df2["city"].fillna("-1", inplace = True)

def int_education_status(status):
    if status == "Candidate of Sciences":
        return 9
    if status == "PhD":
        return 8
    if status == "Alumnus (Specialist)":
        return 7
    if status == "Student (Specialist)":
        return 6
    if status == "Alumnus (Master's)":
        return 5
    if status == "Student (Master's)":
        return 4
    if status == "Alumnus (Bachelor's)":
        return 3
    if status == "Student (Bachelor's)":
        return 2
    if status == "Undergraduate applicant":
        return 1


df["education_status"] = df["education_status"].apply(int_education_status)

df2["education_status"] = df2["education_status"].apply(int_education_status)

'''
print(df.groupby(by = "city")["result"].mean())
print(df.groupby(by = "langs")["result"].mean())

print(df.groupby(by = "relation")["result"].mean())
print(df.groupby(by = "people_main")["result"].mean())
print(df.groupby(by = "life_main")["result"].mean())
print(df.groupby(by = "education_status")["result"].mean())

print(df.groupby(by = "sex")["result"].mean())
print(df.groupby(by = "occupation_type")["result"].mean())
'''
def get_russian(lang):
    if lang.find("Русский")>=0:
        return 1
    return 0
def get_english(lang):
    if lang.find("English")>=0:
        return 1
    return 0
def get_others(lang):
    if lang.find("Русский")<0 and lang.find("English")<0:
        return 1
    return 0

df["Russian"] = df["langs"].apply(get_russian)
df["English"] = df["langs"].apply(get_english)
df["Other_langs"] = df["langs"].apply(get_others)

df2["Russian"] = df2["langs"].apply(get_russian)
df2["English"] = df2["langs"].apply(get_english)
df2["Other_langs"] = df2["langs"].apply(get_others)

def false_to_0(f):
    if f == "False":
        return 0
    return f


df["life_main"] = df["life_main"].apply(false_to_0)
df["life_main"] = df["life_main"].apply(int)

df["people_main"] = df["people_main"].apply(false_to_0)
df["people_main"] = df["people_main"].apply(int)

df2["life_main"] = df2["life_main"].apply(false_to_0)
df2["life_main"] = df2["life_main"].apply(int)

df2["people_main"] = df2["people_main"].apply(false_to_0)
df2["people_main"] = df2["people_main"].apply(int)


def get_moscow(city):
    if city.find("Moscow")>=0:
        return 1
    return 0
def get_saint_Petersburg(city):
    if city.find("Saint Petersburg")>=0:
        return 1
    return 0
def get_other_cities(city):
    if city.find("Moscow")<0 and city.find("Saint Petersburg")<0:
        return 1
    return 0


df["Moscow"] = df["city"].apply(get_moscow)
df["Saint Petersburg"] = df["city"].apply(get_saint_Petersburg)
df["Other_cities"] = df["city"].apply(get_other_cities)

df2["Moscow"] = df2["city"].apply(get_moscow)
df2["Saint Petersburg"] = df2["city"].apply(get_saint_Petersburg)
df2["Other_cities"] = df2["city"].apply(get_other_cities)

def occupation_type_int(oc_type):
    if oc_type == "university":
        return 1
    if oc_type == "work":
        return 0
    return -1

df["occupation_type"] = df["occupation_type"].apply(occupation_type_int)
df2["occupation_type"] = df2["occupation_type"].apply(occupation_type_int)

df.drop(["langs", "city", "has_photo", "has_mobile", "bdate", "education_form", "last_seen", "occupation_name", "career_start", "career_end"], axis = 1, inplace = True)
df2.drop(["langs", "city", "has_photo", "has_mobile", "bdate", "education_form", "last_seen", "occupation_name", "career_start", "career_end"], axis = 1, inplace = True)


x_train = df.drop("result", axis = 1)
y_train = df["result"]

x_test = df2

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 11)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

ID = df2["id"]

result = pd.DataFrame({"id":ID, 'result':y_pred})
result.to_csv('answer.csv', index = False)