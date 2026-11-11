//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(associated_type_defaults)]

// Tests that `Self` is not assumed to implement `Sized` when used as an
// associated type default.

trait Inner<S> {}

trait MyTrait {
    type This = Self;  //~ error: size for values of type `Self` cannot be known
    fn something<I: Inner<Self::This>>(i: I);
}

fn main() {}
