import sys
from bs4 import BeautifulSoup

html = open(sys.argv[1], 'r').read()
soup = BeautifulSoup(html, features="html.parser")
# The tables are:
# Tier 1                    <-- this is already checked by main CI, so we ignore it here
# Tier 2 with host tools    <-- we want this one
# Tier 2 without host tools <-- and also this
# Tier 3
for table in soup.find_all("table")[1:3]:
    for row in table.find_all('tr'):
        code = row.find('code')
        if code is not None:
            print(code.text)
