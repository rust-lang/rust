//! Test that nonsense bounds prevent consts from being evaluated at all.
//@ check-pass

#![feature(generic_const_items)]
#![allow(incomplete_features)]
trait Trait {
    const ASSOC: u32;
}

// rustfmt eats the where bound
#[rustfmt::skip]
const ASSOC: u32 = <&'static ()>::ASSOC where for<'a> &'a (): Trait;

fn main() {}
