# A little helper script for passing rustdoc output through markdown
#  and pandoc

import sys, os, commands;

if len(sys.argv) < 2:
    print("Please provide an input crate")
    sys.exit(1)

crate = sys.argv[1]

status, output = commands.getstatusoutput("rustdoc " + crate)

basename = os.path.splitext(os.path.basename(crate))[0]

markdownfile = basename + ".md"

f = open(markdownfile, 'w')
f.write(output)
f.close()

status, output = commands.getstatusoutput("markdown " + markdownfile)

htmlfile = basename + ".md.html"

f = open(htmlfile, 'w')
f.write(output)
f.close()

pdcmd = "pandoc --standalone --toc --section-divs --number-sections \
         --from=markdown --to=html --css=rust.css \
         --output=" + basename + ".pd.html " + markdownfile

os.system(pdcmd)
