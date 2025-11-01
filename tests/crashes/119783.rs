//@ known-bug: #119783
#![feature(associated_const_equality, min_generic_const_args)]

trait Trait {
    #[type_const]
    const F: fn();
}

fn take(_: impl Trait<F = { || {} }>) {}

fn main() {}
