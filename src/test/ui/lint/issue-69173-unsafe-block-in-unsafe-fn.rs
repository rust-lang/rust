// Regression test for #69173: do not warn for `unsafe` blocks in `unsafe` functions

// check-pass

#![allow(dead_code)]
#![deny(unused_unsafe)]

unsafe fn foo() {}
unsafe fn bar() { unsafe { foo(); } }

fn main() {}
