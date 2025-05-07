//@ check-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// Issue #1821 - Don't recurse trying to typecheck this



enum t {
    foo(Vec<t>)
}
pub fn main() {}
