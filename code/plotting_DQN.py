import matplotlib.pyplot as plt

FILEPATH = "output final code.txt"

with open(FILEPATH, 'r') as f:
    lines = f.read().split("\n")

scores = list()
for line in lines[1:-1:2]:
    print(line)
    score = int(float(line.split("the reward is ")[1].split(" ")[0])) * (-1)
    print(score)
    scores.append(score)

print(len(scores))


plt.figure(figsize=(8, 4))
plt.plot(scores)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.savefig("episode-score.png")