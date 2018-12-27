// compile-flags: -Z parse-only

enum e = isize; //~ ERROR expected one of `<`, `where`, or `{`, found `=`
