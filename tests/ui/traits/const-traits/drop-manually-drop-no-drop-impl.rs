//@[new] compile-flags: -Znext-solver
//@ revisions: old new
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

use std::mem::ManuallyDrop;

struct Moose;

impl Drop for Moose {
    fn drop(&mut self) {}
}

struct ConstDropper<T>(ManuallyDrop<T>);

const fn foo(_var: ConstDropper<Moose>) {}

fn main() {}
