//@[new] compile-flags: -Znext-solver
//@ revisions: old new

#![feature(const_destruct)]
#![feature(const_trait_impl)]

use std::mem::ManuallyDrop;

struct Moose;

impl Drop for Moose {
    fn drop(&mut self) {}
}

struct ConstDropper<T>(ManuallyDrop<T>);

impl<T> const Drop for ConstDropper<T> {
    fn drop(&mut self) {}
}

const fn foo(_var: ConstDropper<Moose>) {}
//~^ ERROR destructor of `ConstDropper<Moose>` cannot be evaluated at compile-time

fn main() {}
