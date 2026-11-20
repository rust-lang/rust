//@[new] compile-flags: -Znext-solver
//@ revisions: old new
//@ check-pass

#![feature(const_destruct)]
#![feature(const_trait_impl)]

use std::mem::ManuallyDrop;

struct Moose;

impl Drop for Moose {
    fn drop(&mut self) {}
}

struct ConstDropper<T>(ManuallyDrop<T>);

const impl<T> Drop for ConstDropper<T> {
    fn drop(&mut self) {}
}

const fn foo(_var: ConstDropper<Moose>) {}

fn main() {}
