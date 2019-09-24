// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// Issue #1821 - Don't recurse trying to typecheck this


// pretty-expanded FIXME #23616

enum t {
    foo(Vec<t>)
}
pub fn main() {}
