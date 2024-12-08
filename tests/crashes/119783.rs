//@ known-bug: #119783
#![feature(associated_const_equality)]

trait Trait { const F: fn(); }

fn take(_: impl Trait<F = { || {} }>) {}

fn main() {}
