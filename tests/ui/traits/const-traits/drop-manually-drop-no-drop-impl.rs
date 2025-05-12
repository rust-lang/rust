//@[new] compile-flags: -Znext-solver
//@ revisions: old new
//@ check-pass

use std::mem::ManuallyDrop;

struct Moose;

impl Drop for Moose {
    fn drop(&mut self) {}
}

struct ConstDropper<T>(ManuallyDrop<T>);

const fn foo(_var: ConstDropper<Moose>) {}

fn main() {}
