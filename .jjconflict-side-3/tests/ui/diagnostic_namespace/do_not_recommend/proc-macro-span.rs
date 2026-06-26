//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: --crate-type=lib
//@[next] compile-flags: -Znext-solver
//@ edition: 2024
//@ proc-macro: proc_macro_repro.rs

// Regression test for https://github.com/rust-lang/rust/issues/156759.

extern crate proc_macro_repro;

struct Arg(std::marker::PhantomPinned);

#[proc_macro_repro::repro]
fn f(arg: &mut Arg);
//~^ ERROR mutable reference to C++ type requires a pin
