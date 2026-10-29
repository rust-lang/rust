//@ proc-macro: builtin-attrs.rs
//@ compile-flags:--test
//@ ignore-backends: gcc

#![feature(decl_macro, test)]

extern crate test;
extern crate builtin_attrs;
use builtin_attrs::{test, bench};

#[test] // OK, shadowed
fn test() {}

#[bench] // OK, shadowed
fn bench(b: &mut test::Bencher) {}

fn not_main() {
    Test;
    Bench;
    NonExistent; //~ ERROR cannot find value `NonExistent` in this scope
}
