// compile-pass
// pretty-expanded FIXME(#23616)

// Issue #1821: don't recurse trying to type-check this.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

enum t {
    foo(Vec<t>)
}

pub fn main() {}
