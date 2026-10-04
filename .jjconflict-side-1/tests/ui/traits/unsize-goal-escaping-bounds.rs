//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@ ignore-compare-mode-next-solver (explicit revisions)

#![feature(unsize)]

use std::marker::Unsize;

trait Trait {}
impl Trait for () {}

fn foo()
where
    for<'a> (): Unsize<dyn Trait + 'a>,
{
}

fn main() {
    foo();
    //[current]~^ ERROR the trait bound `for<'a> (): Unsize<(dyn Trait + 'a)>` is not satisfied
}
