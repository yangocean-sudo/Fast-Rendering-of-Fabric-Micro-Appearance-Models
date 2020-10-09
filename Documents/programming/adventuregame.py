#-*- codeing = utf -8 -*-
#@Time : 2020/10/5 9:22 下午
#@Author : Cui Yangyang
#@File : adventure game.py
#@Software: PyCharm

print("Welcome to my game!")

playerName = input("What is your name? ")

print("Hello, " + playerName)


print("Choose a character: ", "1. Knight", "2. Warrior",
      "3. Wizard", sep= '\n')

characterList = ["Knight", "Warrior", "Wizzard"]
character = int(input("Type in your character number: "))
print("You chose: " + characterList[character - 1])