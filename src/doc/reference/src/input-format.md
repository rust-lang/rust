# Input format

Rust input is interpreted as a sequence of Unicode code points encoded in UTF-8.
Most Rust grammar rules are defined in terms of printable ASCII-range
code points, but a small number are defined in terms of Unicode properties or
explicit code point lists. [^inputformat]

[^inputformat]: Substitute definitions for the special Unicode productions are
  provided to the grammar verifier, restricted to ASCII range, when verifying the
  grammar in this document.
