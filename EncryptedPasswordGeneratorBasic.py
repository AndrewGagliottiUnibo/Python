import string
import random

# Insert an input seed to begin the calculation
seed = input("Enter an alphanumerical seed: ")

# Encryption Algorithm - basic version
encrypter = []

for i in range(0, len(seed)):
    for letter in seed:
        if letter.isnumeric():
            encrypter.append(random.choice(list(string.ascii_letters)))
        if letter.isalpha():
            encrypter.append(random.choice(list(string.digits)))
        if letter.isspace():
            encrypter.append(random.choice(list(string.ascii_letters)))

    random.shuffle(encrypter)

random.shuffle(encrypter)

# Now due to the fact that the shuffle has increased the list size we need to reestablish the correct dimension of it,
# based on the seed length. After that we can print the password.
password = ""
for i in range(0, int(len(encrypter)/len(seed))):
    password += str(encrypter[i])

print("Generated password is: " + password)
