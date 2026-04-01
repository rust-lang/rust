// Unused assignments to an unused variable should trigger only the `unused_variables` lint and not
// also the `unused_assignments` lint.  This test covers the situation where the span of the unused
// variable identifier comes from a different scope to the binding pattern - here, from a proc
// macro's input tokenstream (whereas the binding pattern is generated within the proc macro
// itself).
//
// Regression test for https://github.com/rust-lang/rust/issues/151514
//
//@ check-pass
//@ proc-macro: unused_assignment_proc_macro.rs
#![warn(unused)]

extern crate unused_assignment_proc_macro;
use unused_assignment_proc_macro::Drop;

#[derive(Drop)]
pub struct S {
    a: (), //~ WARN unused variable: `a`
}

fn main() {}
