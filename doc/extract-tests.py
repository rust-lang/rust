# Script for extracting compilable fragments from markdown
# documentation. See prep.js for a description of the format
# recognized by this tool. Expects a directory fragements/ to exist
# under the current directory, and writes the fragments in there as
# individual .rs files.

import sys, re;

if len(sys.argv) < 2:
    print("Please provide an input filename")
    sys.exit(1)

filename = sys.argv[1]
f = open(filename)
lines = f.readlines()
f.close()

cur = 0
line = ""
chapter = ""
chapter_n = 0

while cur < len(lines):
    line = lines[cur]
    cur += 1
    chap = re.match("# (.*)", line);
    if chap:
        chapter = re.sub(r"\W", "_", chap.group(1)).lower()
        chapter_n = 1
    elif re.match("~~~", line):
        block = ""
        ignore = False
        while cur < len(lines):
            line = lines[cur]
            cur += 1
            if re.match(r"\s*## (notrust|ignore)", line):
                ignore = True
            elif re.match("~~~", line):
                break
            else:
                block += re.sub("^# ", "", line)
        if not ignore:
            if not re.search(r"\bfn main\b", block):
                if re.search(r"(^|\n) *(native|use|mod|import|export)\b", block):
                    block += "\nfn main() {}\n"
                else:
                    block = "fn main() {\n" + block + "\n}\n"
            if not re.search(r"\buse std\b", block):
                block = "use std;\n" + block;
            filename = "fragments/" + str(chapter) + "_" + str(chapter_n) + ".rs"
            chapter_n += 1
            f = open(filename, 'w')
            f.write(block)
            f.close()

