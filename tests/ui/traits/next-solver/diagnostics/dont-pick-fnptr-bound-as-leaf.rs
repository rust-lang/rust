//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// When emitting an error for `Foo: Trait` not holding we attempt to find a nested goal
// to give as the reason why the bound does not hold. This test checks that we do not
// try to tell the user that `Foo: FnPtr` is unimplemented as that would be confusing.

#![feature(fn_ptr_trait)]

use std::marker::FnPtr;

trait Trait {}

impl<T: FnPtr> Trait for T {}

struct Foo; //~ HELP: the trait `Trait` is not implemented for `Foo`

fn requires_trait<T: Trait>(_: T) {}
//~^ NOTE: required by a bound in `requires_trait`
//~| NOTE: required by this bound in `requires_trait`

fn main() {
    requires_trait(Foo);
    //~^ ERROR: the trait bound `Foo: Trait` is not satisfied
    //~| NOTE: unsatisfied trait bound
    //~| NOTE: required by a bound introduced by this call
}
