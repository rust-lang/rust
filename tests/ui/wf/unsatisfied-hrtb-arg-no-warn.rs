//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
//@ edition: 2021

// Do not emit the UNSATISFIED_HRTB_ARG FCW for perfectly fine code.

#![allow(dead_code)]

fn f<T>(_: impl for<'a> Fn(&'a T)) {}

fn g<T>(_: &T) where for<'a> &'a T: Copy {}

fn main() {}
