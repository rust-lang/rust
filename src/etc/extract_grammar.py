#!/usr/bin/env python

# This script is for extracting the grammar from the rust docs.

import fileinput

collections = { "gram": [],
                "keyword": [],
                "reserved": [],
                "binop": [],
                "unop": [] }


in_coll = False
coll = ""

for line in fileinput.input(openhook=fileinput.hook_encoded("utf-8")):
    if in_coll:
        if line.startswith("~~~~"):
            in_coll = False
        else:
            if coll in ["keyword", "reserved", "binop", "unop"]:
                for word in line.split():
                    if word not in collections[coll]:
                        collections[coll].append(word)
            else:
                collections[coll].append(line)

    else:
        if line.startswith("~~~~"):
            for cname in collections:
                if ("." + cname) in line:
                    coll = cname
                    in_coll = True
                    break

# Define operator symbol-names here

tokens = ["non_star", "non_slash", "non_eol",
          "non_single_quote", "non_double_quote", "ident" ]

symnames = {
".": "dot",
"+": "plus",
"-": "minus",
"/": "slash",
"*": "star",
"%": "percent",

"~": "tilde",
"@": "at",

"!": "not",
"&": "and",
"|": "or",
"^": "xor",

"<<": "lsl",
">>": "lsr",
">>>": "asr",

"&&": "andand",
"||": "oror",

"<" : "lt",
"<=" : "le",
"==" : "eqeq",
">=" : "ge",
">" : "gt",

"=": "eq",

"+=": "plusequal",
"-=": "minusequal",
"/=": "divequal",
"*=": "starequal",
"%=": "percentequal",

"&=": "andequal",
"|=": "orequal",
"^=": "xorequal",

">>=": "lsrequal",
">>>=": "asrequal",
"<<=": "lslequal",

"::": "coloncolon",

"//": "linecomment",
"/*": "openblockcomment",
"*/": "closeblockcomment"
}

lines = []

for line in collections["gram"]:
    line2 = ""
    for word in line.split():
        # replace strings with keyword-names or symbol-names from table
        if word.startswith("\""):
            word = word[1:-1]
            if word in symnames:
                word = symnames[word]
            else:
                for ch in word:
                    if not ch.isalpha():
                        raise Exception("non-alpha apparent keyword: "
                                        + word)
                if word not in tokens:
                    if (word in collections["keyword"] or
                        word in collections["reserved"]):
                       tokens.append(word)
                    else:
                        raise Exception("unknown keyword/reserved word: "
                                        + word)

        line2 += " " + word
    lines.append(line2)


for word in collections["keyword"] + collections["reserved"]:
    if word not in tokens:
        tokens.append(word)

for sym in collections["unop"] + collections["binop"] + symnames.keys():
    word = symnames[sym]
    if word not in tokens:
        tokens.append(word)


print("%start parser, token;")
print("%%token %s ;" % ("\n\t, ".join(tokens)))
for coll in ["keyword", "reserved"]:
    print("%s: %s ; " % (coll, "\n\t| ".join(collections[coll])));
for coll in ["binop", "unop"]:
    print("%s: %s ; " % (coll, "\n\t| ".join([symnames[x]
                                              for x in collections[coll]])));
print("\n".join(lines));
