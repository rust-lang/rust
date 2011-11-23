#!/usr/bin/env python

import sys
import os.path

def scrub(b):
  if sys.version_info >= (3,) and type(b) == bytes:
    return b.decode('ascii')
  else:
    return b

src_dir = scrub(os.getenv("CFG_SRC_DIR"))
if not src_dir:
  raise Exception("missing env var CFG_SRC_DIR")


def get_keywords():
  keywords_file = os.path.join(src_dir, "doc", "keywords.txt")
  keywords = []
  for line in open(keywords_file).readlines():
    if not line or line.startswith('#'):
      continue
    for kw in line.split():
      if kw.isalnum():
        keywords.append(kw)
  return keywords


def sort(keywords, ncols):
  """Sort keywords in a column-major ordered table.

  Args:
    keywords: List of keywords
    ncols: Number of columns to be sorted
  """
  ## sort and remove duplicates
  keywords = sorted(list(set(keywords)))
  sz = len(keywords)

  if sz % ncols > 0:
    nrows = sz / ncols + 1
  else:
    nrows = sz / ncols

  result = []
  max = ncols * nrows
  for i in xrange(0, max):
    result.append("")

  for i in xrange(1, sz+1):
    if i % nrows == 0:
      extra = 0
    else:
      extra = 1
    pos = (((i + (nrows - 1)) % nrows) * ncols) + \
          (i / nrows + extra)
    result[pos - 1] = keywords[i - 1]

  return rows(result, ncols)


def rows(keywords, ncols):
  """Split input list of keywords into rows.

  Each contains ncols or ncols-1 elements.

  Args:
    keywords: List of keywords sorted in column-major order
    ncols: Number of columns
  """
  sz = len(keywords)
  result = []
  i = 0
  while i < sz:
    if i + ncols < sz:
      se = i + ncols
    else:
      se = sz
    result.append(keywords[i:se])
    i = se
  return result


def table(rows):
  """Render rows in a texinfo multitable."""
  result = ["@multitable @columnfractions .15 .15 .15 .15 .15\n"]
  for row in rows:
    result += ["@item @code{" + row[0] + "}\n"];
    for e in row[1:]:
      result += ["@tab @code{" + e + "}\n"];
  result += ["@end multitable\n"];
  return result


def main(oargs):
  keywords = get_keywords()
  out_file = open(os.path.join("doc", "keywords.texi"), 'w')
  for line in table(sort(keywords, 5)):
    out_file.write(line)
  out_file.close()

if __name__ == '__main__':
  main(sys.argv[1:])
