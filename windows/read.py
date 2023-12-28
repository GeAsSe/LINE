import json

if __name__ == "__main__":
    print("Hello World!")

f = open('pokemon.json', 'r')
plist = json.load(f)
print(len(plist))