import sys

for i in range(1,len(sys.argv)):
  content = open('test.template').read()
  content = content.replace("@NAME@", sys.argv[i])
  if sys.argv[i].startswith('FAIL_'):
    content = content.replace("@EXPECTFAIL@", "; XFAIL: *")
  else:
    content = content.replace("@EXPECTFAIL@", "")
  open('./testfiles/'+sys.argv[i]+".test", 'w+').write(content)
