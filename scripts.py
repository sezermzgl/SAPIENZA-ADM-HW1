# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input())
    if n % 2 != 0:
        print("Weird")
    elif n % 2 == 0:
        if 2 <= n <= 5:
            print("Not Weird")
        elif 6 <= n <= 20:
            print("Weird")
        elif n > 20:
            print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    result_float = int(a/b)
    result_int = float(a/b)
    print(result_float)
    print(result_int)

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

# Write a function
def is_leap(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False
            

# Print Function
if __name__ == '__main__':
    n = int(input())
    for a in range(1, n + 1):
        print(a, end="")

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    coordinates = [[i, j, k] 
    for i in range(x + 1)
    for j in range(y + 1)
    for k in range(z + 1) 
 
if i + j + k != n]
print(coordinates)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    sorted_set = sorted(set(arr))
    print(sorted_set[-2])

# Nested Lists
if __name__ == '__main__':
    n = int(input())
    records = []
    for _ in range(n):
        name = input()
        grade = float(input())
        records.append([name, grade])
    grades = sorted(set([record[1] for record in records]))
    nd_lowest_grade = grades[1]
    
    nd_lowest_students = [record[0] for record in records if record[1] == nd_lowest_grade]
    
    for std in sorted(nd_lowest_students):
        print(std)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    avrg = sum(student_marks[query_name]) / len(student_marks[query_name])
    
    print(f"{avrg:.2f}")

# Lists
if __name__ == '__main__':
    n = int(input())
    lst = []          
    
    for i in range(n):
        command = input().split()  
        if command[0] == "insert":
            lst.insert(int(command[1]), int(command[2]))
        elif command[0] == "print":
            print(lst)
        elif command[0] == "remove":
            lst.remove(int(command[1]))
        elif command[0] == "append":
            lst.append(int(command[1]))
        elif command[0] == "sort":
            lst.sort()
        elif command[0] == "pop":
            lst.pop()
        elif command[0] == "reverse":
            lst.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    t = tuple(map(int, input().split()))
    tuple_hash = hash(t)
    print(tuple_hash)

# sWAP cASE
def swap_case(sw):
    swapped_sw = sw.swapcase()
    return swapped_sw

# String Split and Join

def split_and_join(line):
    line_modified = line.replace(" ", "-")
    return line_modified
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    lst = list(string)
    lst[position] = character
    string = ''.join(lst)
    return string

# Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i + len(sub_string)] == sub_string:
            count += 1
    return count

# Introduction to Sets
def average(array):
    distinct_heights = set(arr)
    return round(sum(distinct_heights) / len(distinct_heights), 3)

# Symmetric Difference
m = int(input())  
set_m = set(map(int, input().split()))  
n = int(input())  
set_n = set(map(int, input().split())) 
symmetric_difference = sorted(set_m.symmetric_difference(set_n))
for value in symmetric_difference:
    print(value)

# No Idea!
n, m = map(int, input().split()) 
array = list(map(int, input().split()))  
A = set(map(int, input().split()))  
B = set(map(int, input().split())) 

happiness = 0
for element in array:
    if element in A:
        happiness += 1
    elif element in B:
        happiness -= 1
print(happiness)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
distinct_countries = set()
for _ in range(N):
    country = input().strip()
    distinct_countries.add(country)
print(len(distinct_countries))

# Set .discard(), .remove() & .pop()
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s = list(map(int, input().split()))
s = set(s[::-1])
c = int(input())
for i in range(c):
    l = input().split()
    if l[0] == 'pop':
        try:
            s.pop()
        except:
            None
    elif l[0] == 'remove':
        try:
            s.remove(int(l[1]))
        except:
            None
    else:
       s.discard(int(l[1]))
    
print(sum(s))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())  
english_subscribers = set(map(int, input().split()))  
b = int(input())  
french_subscribers = set(map(int, input().split()))  
total_subscribers = english_subscribers.union(french_subscribers)
print(len(total_subscribers))

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
english_count = int(input())
english_roll_numbers = set(map(int, input().split()))
french_count = int(input())
french_roll_numbers = set(map(int, input().split()))
both_subscribed = english_roll_numbers.intersection(french_roll_numbers)
print(len(both_subscribed))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
english_count = int(input())
english_roll_numbers = set(map(int, input().split()))
french_count = int(input())
french_roll_numbers = set(map(int, input().split()))
only_english = english_roll_numbers.difference(french_roll_numbers)
print(len(only_english))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
english_count = int(input())
english_roll_numbers = set(map(int, input().split()))
french_count = int(input())
french_roll_numbers = set(map(int, input().split()))
unique_subscribers = english_roll_numbers.symmetric_difference(french_roll_numbers)
print(len(unique_subscribers))

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
num_elements_A = int(input())
A = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    operation, _ = input().split()
    
    other_set = set(map(int, input().split()))
    
    getattr(A, operation)(other_set)
print(sum(A))# Enter your code here. Read input from STDIN. Print output to STDOUT

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
K = int(input())
room_numbers = list(map(int, input().split()))
room_count = Counter(room_numbers)
#print(room_count)
for room, count in room_count.items():
    if count == 1:
        print(room)
        break

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for _ in range(T):
    len_A = int(input())
    set_A = set(map(int, input().split()))
    len_B = int(input())
    set_B = set(map(int, input().split()))
    print(set_A.issubset(set_B))

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
is_strict_superset = True
for _ in range(n):
    other_set = set(map(int, input().split()))
    
    if not (A > other_set):  
        is_strict_superset = False
        break
print(is_strict_superset)

# String Validators
if __name__ == '__main__':
    S = input()
    if any(char.isalnum() for char in S):
        print(True)
    else:
        print(False)
    if any(char.isalpha() for char in S):
        print(True)
    else:
        print(False)
    if any(char.isdigit() for char in S):
        print(True)
    else:
        print(False)
    if any(char.islower() for char in S):
        print(True)
    else:
        print(False)
    if any(char.isupper() for char in S):
        print(True)
    else:
        print(False)

# Text Alignment
thickness = int(input()) 
c = 'H'
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))

# Text Wrap

def wrap(string, max_width):
    l = [string[i:i+max_width] for i in range(0,len(string),max_width)]
    return "\n".join(l)

# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, M = map(int, input().split())
for i in range(1, N, 2):  
    pattern = (".|." * i).center(M, "-")
    print(pattern)
print("WELCOME".center(M, "-"))
for i in range(N-2, -1, -2):  
    pattern = (".|." * i).center(M, "-")
    print(pattern)

# String Formatting
def print_formatted(number):
    width = len(bin(number)) - 2
    for i in range(1, number + 1):
        print(str(i).rjust(width), oct(i)[2:].rjust(width), hex(i)[2:].upper().rjust(width), bin(i)[2:].rjust(width))

# Alphabet Rangoli
import string
def print_rangoli(size):
 
    alphabet = string.ascii_lowercase
    width = 4 * size - 3
    pattern = []
    for i in range(size):
        letters = "-".join(alphabet[size-1:i:-1] + alphabet[i:size])
        pattern.append(letters.center(width, "-"))
    print("\n".join(pattern[::-1] + pattern[1:]))

# Capitalize!

# Complete the solve function below.
def solve(s):
    return " ".join([word.capitalize() if word else "" for word in s.split(" ")])

# The Minion Game
def minion_game(string):
    vowels = "AEIOU"
    kev = 0
    stuart = 0
    length = len(string)
    
    for i in range(length):
        if string[i] in vowels:
            kev += length - i
        else:
            stuart += length - i
    if stuart > kev:
        print(f"Stuart {stuart}")
    elif kev > stuart:
        print(f"Kevin {kev}")
    else:
        print("Draw")

# Merge the Tools!
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        substring = string[i:i+k]
        seen = ""
        for char in substring:
            if char not in seen:
                seen += char
        print(seen)

# collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
X = int(input())
shoe_inventory = Counter(map(int, input().split()))
N = int(input())
total_earnings = 0
for _ in range(N):
    size, price = map(int, input().split())
    
    if shoe_inventory[size] > 0:
        total_earnings += price
        shoe_inventory[size] -= 1  
print(total_earnings)

# DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
n, m = map(int, input().split())
group_a_positions = defaultdict(list)
for i in range(1, n + 1):  
    word = input().strip()
    group_a_positions[word].append(i)
for _ in range(m):
    word = input().strip()
    if word in group_a_positions:
        print(" ".join(map(str, group_a_positions[word])))
    else:
        print(-1)

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
columns = input().split()
marks_index = columns.index("MARKS")
total_marks = sum(int(input().split()[marks_index]) for _ in range(N))
avg_marks = total_marks / N
print(f"{avg_marks:.2f}")

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
N = int(input())
items = OrderedDict()
for _ in range(N):
    *item_name, price = input().split()
    item_name = " ".join(item_name)  
    price = int(price)  
    if item_name in items:
        items[item_name] += price
    else:
        items[item_name] = price
for item_name, net_price in items.items():
    print(f"{item_name} {net_price}")

# Word Order
from collections import OrderedDict
n = int(input())
word_count = OrderedDict()
for _ in range(n):
    word = input().strip()  
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1
print(len(word_count))
print(" ".join(map(str, word_count.values())))

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
N = int(input())
d = deque()
for _ in range(N):
    operation = input().split()
    
    if operation[0] == 'append':
        d.append((operation[1]))
    elif operation[0] == 'appendleft':
        d.appendleft((operation[1]))
    elif operation[0] == 'pop':
        d.pop()
    elif operation[0] == 'popleft':
        d.popleft()
print(" ".join(map(str, d)))

# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
T = int(input())

results = []
for _ in range(T):
    n = int(input())
    
    cubes = deque(map(int, input().split()))
    
    current_max = float('inf')
    possible = True
    while cubes:
    
        if cubes[0] >= cubes[-1]:
            choice = cubes.popleft()
        else:
            choice = cubes.pop()
        if choice > current_max:
            possible = False
            break
        
        current_max = choice
    if possible:
        results.append("Yes")
    else:
        results.append("No")
print("\n".join(results))

# Company Logo
from collections import Counter

if __name__ == '__main__':
    a = [b for b in input()]
    s = Counter(a)
    s = [(i, j) for i, j in s.items()]
    s = sorted(s, key=lambda x: (-x[1], x[0]))
    for i in s[:3]:
        print(f"{i[0]} {i[1]}")

# Calendar Module
import calendar
from datetime import datetime
month, day, year = map(int, input().split())
date = datetime(year, month, day)
day_name = calendar.day_name[date.weekday()]
print(day_name.upper())

# Time Delta
import os
from datetime import datetime
def time_delta(t1, t2):
    format_str = "%a %d %b %Y %H:%M:%S %z"
    time1 = datetime.strptime(t1, format_str)
    time2 = datetime.strptime(t2, format_str)
    delta_seconds = abs(int((time1 - time2).total_seconds()))
    return str(delta_seconds)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for _ in range(T):
    a, b = input().split()
    
    try:
        result = int(a) // int(b)
        print(result)
    except ZeroDivisionError as e:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print(f"Error Code: {e}")

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, X = map(int, input().split())
subject_scores = [list(map(float, input().split())) for _ in range(X)]
for student_scores in zip(*subject_scores):
    average_score = sum(student_scores) / X
    print(f"{average_score:.1f}")

# Athlete Sort
N, M = map(int, input().split())
athletes = [list(map(int, input().split())) for _ in range(N)]
K = int(input())
athletes.sort(key=lambda x: x[K])
for athlete in athletes:
    print(" ".join(map(str, athlete)))

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
S = input()
lowercase = sorted([char for char in S if char.islower()])
uppercase = sorted([char for char in S if char.isupper()])
odd_digits = sorted([char for char in S if char.isdigit() and int(char) % 2 != 0])
even_digits = sorted([char for char in S if char.isdigit() and int(char) % 2 == 0])
result = "".join(lowercase + uppercase + odd_digits + even_digits)
print(result)

# Map and Lambda Function
cube = lambda x: x ** 3
def fibonacci(n):
    if n == 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
def is_floating_point(num_str):
    try:
        if num_str.count('.') == 1 and len(num_str) > 1:
            float(num_str)
            return True
        else:
            return False
    except ValueError:
        return False
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        N = input().strip()
        print(is_floating_point(N))

# Re.split()
regex_pattern = r"\W"	# Do not delete 'r'.

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input().strip()
pattern = r'([a-zA-Z0-9])\1'
match = re.search(pattern, s)
print(match.group(1) if match else -1)

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
vowels = r'[AEIOUaeiou]{2,}'
consonants = r'[QWRTYPSDFGHJKLZXCVBNMqwrtpsdfghjklzxcvbnm]'
regex_pattern = fr'(?<={consonants}){vowels}(?={consonants})'
input_string = input()
matches = re.findall(regex_pattern, input_string)
if matches:
    for match in matches:
        print(match)
else:
    print("-1")


# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
k = input()
matches = list(re.finditer(r'(?={})'.format(k), S))
if not matches:
    print("(-1, -1)")
else:
    for match in matches:
        print(f"({match.start()}, {match.start() + len(k) - 1})")

# Regex Substitution
import re
N = int(input())
for _ in range(N):
    line = input()
    
    line = re.sub(r'(?<= )\&\&(?!\S)', 'and', line)
    
    line = re.sub(r'(?<= )\|\|(?!\S)', 'or', line)
    
    print(line)

# Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.

# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for _ in range(n):
    number = input().strip()
    
    if re.match(r'^[789]\d{9}$', number):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
pattern = r'^[a-zA-Z][\w\.\-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$'
for _ in range(n):
    line = input().strip()
    name, email = line.split()
    email = email[1:-1]
    if re.match(pattern, email):
        print(line)

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
css_lines = [input().strip() for _ in range(n)]
pattern = r'#[0-9A-Fa-f]{3}(?!\w)|#[0-9A-Fa-f]{6}(?!\w)'
inside_selector = False
results = []
for line in css_lines:
    if '{' in line:
        inside_selector = True
    elif '}' in line:
        inside_selector = False
    elif inside_selector:
        matches = re.findall(pattern, line)
        results.extend(matches)
for result in results:
    print(result)

# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] else 'None'}")
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] else 'None'}")
n = int(input())
html_code = ""  
for _ in range(n):
    html_code += input().strip() + "\n"
parser = MyHTMLParser()
parser.feed(html_code)

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)
    
    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)
n = int(input())
html_code = ""
for _ in range(n):
    html_code += input().rstrip() + '\n'
parser = MyHTMLParser()
parser.feed(html_code)

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if attrs:
            [print(f'-> {attr[0]} > {attr[1]}') for attr in attrs]
    def handle_startendtag(self, tag, attrs):
        print(tag)
        if attrs:
            [print(f'-> {attr[0]} > {attr[1]}') for attr in attrs]  
  
html = ""       
for _ in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating UID
import re
def validate_uid(uid):
    valid_alpha_count = bool(len(re.findall(r"[A-Z]", uid)) >= 2)
    valid_digit_count = bool(len(re.findall(r"\d", uid)) >= 3)
    alnum_only = bool(len(re.findall(r"[^\w]", uid)) == 0)
    no_repeat = not bool(re.search(r"(\w).*\1", uid))
    valid_char_count = bool(len(uid) == 10)
    return all([valid_alpha_count, valid_digit_count, alnum_only, no_repeat, valid_char_count])
    
testcase_count = int(input())
for _ in range(testcase_count):
    uid = input()
    if validate_uid(uid) == True:
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def validate_credit_card(card_number):
    pattern_1 = r"^[456]\d{3}(-?\d{4}){3}$"
    pattern_2 = r"(\d)\1{3,}" 
    if re.match(pattern_1, card_number):
        if not re.search(pattern_2, card_number.replace("-", "")):
            return "Valid"
    return "Invalid"
N = int(input())
for _ in range(N):
    card_number = input().strip()
    print(validate_credit_card(card_number))

# Validating Postal Codes
regex_integer_in_range = r"^[1-9][0-9]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(\d)\d\1)"	# Do not delete 'r'.

# Matrix Script
import re
n, m = map(int, input().split())
matrix = [input() for _ in range(n)]
decoded_string = "".join([matrix[row][col] for col in range(m) for row in range(n)])
print(re.sub(r'(?<=\w)([^\w]+)(?=\w)', ' ', decoded_string))

# XML 1 - Find the Score

def get_attr_number(node):
    total = 0
    for elem in node.iter():
        score = len(elem.attrib)
        total += score
    return total

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem:etree.Element, level:int) -> None:
    global maxdepth
    level += 1
    maxdepth = max(maxdepth, level)
    for node in elem:
        depth(node, level)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        formatted_numbers = ['+91 ' + number[-10:-5] + ' ' + number[-5:] for number in l]
        f(sorted(formatted_numbers))
    return fun


# Decorators 2 - Name Directory

from operator import itemgetter
def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner

# Arrays

def arrays(arr):
    aa=numpy.array(arr,float)
    return aa[::-1]

# Shape and Reshape
import numpy as np
def reshape_to_3x3(arr):
    np_array = np.array(arr).reshape(3, 3)
    return np_array
input_values = list(map(int, input().split()))
result = reshape_to_3x3(input_values)
print(result)

# Transpose and Flatten
import numpy as np
n, m = map(int, input().split())
matrix = np.array([input().split() for _ in range(n)], int)
print(np.transpose(matrix))
print(matrix.flatten())


# Concatenate
import numpy as np
N, M, P = map(int, input().split())
array_1 = np.array([input().split() for _ in range(N)], int)
array_2 = np.array([input().split() for _ in range(M)], int)
result = np.concatenate((array_1, array_2), axis=0)
print(result)

# Zeros and Ones
import numpy as np
dimensions = tuple(map(int, input().split()))
zeros_array = np.zeros(dimensions, dtype=int)
ones_array = np.ones(dimensions, dtype=int)
print(zeros_array)
print(ones_array)

# Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
print(np.eye(n, m))

# Array Mathematics
import numpy as np
n, m = map(int, input().split())
A = np.array([list(map(int, input().split())) for _ in range(n)])
B = np.array([list(map(int, input().split())) for _ in range(n)])
print(np.add(A, B))
print(np.subtract(A, B))
print(np.multiply(A, B))
print(np.floor_divide(A, B))
print(np.mod(A, B))
print(np.power(A, B))

# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')
arr = np.array(input().split(), float)
print(np.floor(arr))
print(np.ceil(arr))
print(np.rint(arr))

# Sum and Prod
import numpy as np
n, m = map(int, input().split())
array = np.array([input().split() for _ in range(n)], int)
sum_along_axis_0 = np.sum(array, axis=0)
result = np.prod(sum_along_axis_0)
print(result)

# Min and Max
import numpy as np
N, M = map(int, input().split())
array = np.array([input().split() for _ in range(N)], int)
min_values = np.min(array, axis=1)
result = np.max(min_values)
print(result)

# Mean, Var, and Std
import numpy as np
N,M = map(int,input().split())
Na =np.array([list(map(int,input().split())) for _ in range(N)])
print(f"{np.mean(Na,axis=1)}\n{np.var(Na,axis=0)}\n{round(np.std(Na),11)}")

# Dot and Cross
import numpy as np
N = int(input())
A = []
for _ in range(N):
    A.append(list(map(int, input().split())))
B = []
for _ in range(N):
    B.append(list(map(int, input().split())))
A = np.array(A)
B = np.array(B)
result = np.dot(A, B)
print(result)

# Inner and Outer
import numpy as np
A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))
print(np.inner(A, B))
print(np.outer(A, B))

# Polynomials
import numpy as np
P = list(map(float, input().split()))
x = float(input())
print(np.polyval(P, x))

# Linear Algebra
import numpy as np
n = int(input())
ls = []
for i in range(n):
    ls.append(list(map(float, input().split())))
arr = np.array(ls)
print(round(np.linalg.det(arr), 3))

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
def birthdayCakeCandles(candles): 
    m=max(candles) 
    return candles.count(m)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if v1 != v2 and (x2 - x1) % (v1 - v2) == 0 and (x1 - x2) * (v1 - v2) < 0:
        return "YES"
    else:
        return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    liked = [2]
    for i in range(1,n):
        liked.append(math.floor(liked[-1]*3/2))
    return sum(liked)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    return n if k == 1 and n < 10 else superDigit(k * sum(int(d) for d in list(str(n))), 1)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    # Write your code here
    for i in range(n - 1, 0, -1):
        val = arr[i]
        j = i - 1
        while j >= 0 and val < arr[j]:
            arr[j+1] = arr[j]
            print(*arr)
            j -= 1
        arr[j + 1] = val
    print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        print(" ".join(map(str, arr)))
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

