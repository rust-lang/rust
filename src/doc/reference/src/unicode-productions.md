# Unicode productions

A few productions in Rust's grammar permit Unicode code points outside the
ASCII range. We define these productions in terms of character properties
specified in the Unicode standard, rather than in terms of ASCII-range code
points. The grammar has a [Special Unicode Productions][unicodeproductions]
section that lists these productions.

[unicodeproductions]: ../grammar.html#special-unicode-productions
