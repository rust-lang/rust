// This test checks that helper attributes of a derive proc macro can be used together with
// other built-in derive macros.
// issue: rust-lang/rust#132561
//@ check-pass
//@ proc-macro: helper-attr.rs
//@ edition:2021

#[macro_use]
extern crate helper_attr;

use helper_attr::WithHelperAttr;

#[derive(WithHelperAttr, Debug, Clone, PartialEq)]
struct MyStruct<#[x] 'a, #[x] const A: usize, #[x] B> {
    #[x]
    field: &'a [B; A],
}

fn main() {}
