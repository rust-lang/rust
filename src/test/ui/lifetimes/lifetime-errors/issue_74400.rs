//! Regression test for #74400: Type mismatch in function arguments E0631, E0271 are falsely
//! recognized as E0308 mismatched types.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

use std::convert::identity;

fn main() {}

fn f<T, S>(data: &[T], key: impl Fn(&T) -> S) {
}

fn g<T>(data: &[T]) {
    f(data, identity)
    //[base]~^ ERROR implementation of `FnOnce` is not general
    //[nll]~^^ ERROR the parameter type
    //[nll]~| ERROR mismatched types
    //[nll]~| ERROR implementation of `FnOnce` is not general
}
