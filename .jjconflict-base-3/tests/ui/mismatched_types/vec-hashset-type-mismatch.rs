//! regression test for <https://github.com/rust-lang/rust/issues/24819>
//@ dont-require-annotations: NOTE

use std::collections::HashSet;

fn main() {
    let mut v = Vec::new();
    foo(&mut v);
    //~^ ERROR mismatched types
    //~| NOTE expected `&mut HashSet<u32>`, found `&mut Vec<_>`
}

fn foo(h: &mut HashSet<u32>) {}
