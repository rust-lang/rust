# xfail-license

# Script for extracting compilable fragments from markdown
# documentation. See prep.js for a description of the format
# recognized by this tool. Expects a directory fragments/ to exist
# under the current directory, and writes the fragments in there as
# individual .rs files.

import sys, re

if len(sys.argv) < 3:
    print("Please provide an input filename")
    sys.exit(1)

filename = sys.argv[1]
dest = sys.argv[2]
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
    chap = re.match("# (.*)", line)
    if chap:
        chapter = re.sub(r"\W", "_", chap.group(1)).lower()
        chapter_n = 1
    elif re.match("~~~", line):
        # Parse the tags that open a code block in the pandoc format:
        # ~~~ {.tag1 .tag2}
        tags = re.findall("\.([\w-]*)", line)
        block = ""
        ignore = "notrust" in tags or "ignore" in tags
        # Some tags used by the language ref that indicate not rust
        ignore |= "ebnf" in tags
        ignore |= "abnf" in tags
        ignore |= "keyword" in tags
        ignore |= "field" in tags
        ignore |= "precedence" in tags
        xfail = "xfail-test" in tags
        while cur < len(lines):
            line = lines[cur]
            cur += 1
            if re.match("~~~", line):
                break
            else:
                # Lines beginning with '# ' are turned into valid code
                line = re.sub("^# ", "", line)
                # Allow ellipses in code snippets
                line = re.sub("\.\.\.", "", line)
                block += line
        if not ignore:
            if not re.search(r"\bfn main\b", block):
                block = "fn main() {\n" + block + "\n}\n"
            if not re.search(r"\bextern mod extra\b", block):
                block = "extern mod extra;\n" + block
            block = """#[ deny(warnings) ];
#[ allow(unused_variable) ];\n
#[ allow(dead_assignment) ];\n
#[ allow(unused_mut) ];\n
""" + block
            if xfail:
                block = "// xfail-test\n" + block
            filename = (dest + "/" + str(chapter)
                        + "_" + str(chapter_n) + ".rs")
            chapter_n += 1
            f = open(filename, 'w')
            f.write(block)
            f.close()
