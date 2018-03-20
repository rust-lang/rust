// aux-build:builtin-attrs.rs

#![feature(decl_macro)] //~ ERROR `feature` is ambiguous

extern crate builtin_attrs;
use builtin_attrs::{test, bench};
use builtin_attrs::*;

#[repr(C)] //~ ERROR `repr` is ambiguous
struct S;
#[cfg_attr(all(), repr(C))] //~ ERROR `repr` is ambiguous
struct SCond;

#[test] // OK, shadowed
fn test() {}

#[bench] // OK, shadowed
fn bench() {}

fn non_macro_expanded_location<#[repr(C)] T>() { //~ ERROR `repr` is ambiguous
    match 0u8 {
        #[repr(C)] //~ ERROR `repr` is ambiguous
        _ => {}
    }
}

fn main() {
    Test;
    Bench;
    NonExistent; //~ ERROR cannot find value `NonExistent` in this scope
}
