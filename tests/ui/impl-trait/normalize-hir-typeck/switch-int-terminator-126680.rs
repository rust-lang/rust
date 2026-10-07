//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: trait-system-refactor-initiative#303
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #126680
//@ compile-flags: -Zvalidate-mir
//@ edition:2021

// This ICE'd with the old solver as MIR build resulted in an opaque type
// as the argument to a `SwitchInt` terminator. Fixed by normalizing opaque
// types in their defining scope. Regression test for #126680. This should
// compile.

#![feature(type_alias_impl_trait)]
type Bar = impl Sized;

use std::path::Path;

struct Struct {
    pub func: fn(check: Bar),
}

#[define_opaque(Bar)]
fn foo() -> Struct {
    Struct {
        func: |check| if check { () } else { () },
    }
}

fn main() {}
