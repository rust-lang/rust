//@aux-build:../../ui/auxiliary/proc_macro_derive.rs
#![warn(clippy::undocumented_may_panic_call)]

extern crate proc_macro_derive;
use proc_macro_derive::DeriveMayPanic;

fn main() {
    let mut v = vec![1, 2, 3];

    v.push(4);
    //~^ undocumented_may_panic_call

    // Panic: The capacity is less than isize::MAX, so we are safe!
    v.push(5);
}

// Test proc macro derives with configured may-panic function (Vec::push)

#[derive(DeriveMayPanic)]
//~^ undocumented_may_panic_call
struct TestProcMacroNoComment;

// Panic: This derive macro is totally safe for this struct!
#[derive(DeriveMayPanic)]
struct TestProcMacroWithComment;
