//@ revisions: good bad good_new bad_new
//@[good_new] compile-flags: -Znext-solver
//@[bad_new] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[good] check-pass
//@[good_new] check-pass

#![feature(trivial_bounds)]
#![allow(trivial_bounds)]

trait Bad {
    type Assert
    where
        Self: Sized;
}

struct Tail([()]);

impl Bad for Tail {}

#[cfg(any(bad, bad_new))]
const FOO: <Tail as Bad>::Assert = todo!();
//[bad]~^ ERROR the size for values of type `[()]` cannot be known at compilation time
//[bad]~| ERROR the size for values of type `[()]` cannot be known at compilation time
//[bad_new]~^^^ ERROR the size for values of type `[()]` cannot be known at compilation time
//[bad_new]~| ERROR the size for values of type `[()]` cannot be known at compilation time

#[cfg(any(good, good_new))]
// Well-formed in trivially false param-env
fn foo() where Tail: Sized {
    let _: <Tail as Bad>::Assert;
}

fn main() {}
