//! Regression test for #160553
//!
//! Ensure that providing an array const arg with the wrong number of elements
//! when the expected length is a generic associated const doesn't ICE or cause UB.
//@ compile-flags: -Copt-level=0
#![allow(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args, macroless_generic_const_args)]
#![feature(generic_const_parameter_types)]

trait Trait {
    type const LEN: usize;
}

struct S;
impl Trait for S {
    type const LEN: usize = 2;
}

struct Three;
impl Trait for Three {
    type const LEN: usize = 3;
}

fn foo<T: Trait, const A: [u8; <T as Trait>::LEN]>() -> [u8; <T as Trait>::LEN] {
    A
}

// Oversized array in generic context
fn bar_oversized<T: Trait>() -> [u8; <T as Trait>::LEN] {
    foo::<T, { [1, 2, 3] }>()
    //~^ ERROR: the constant `*b"\x01\x02\x03"` is not of type `[u8; <T as Trait>::LEN]`
}

// Undersized array in generic context
fn bar_undersized<T: Trait>() -> [u8; <T as Trait>::LEN] {
    foo::<T, { [] }>()
    //~^ ERROR: the constant `*b""` is not of type `[u8; <T as Trait>::LEN]`
}

// Valid array matching concrete associated const length
fn bar_valid<T: Trait>() -> [u8; 3] {
    foo::<Three, { [1, 2, 3] }>()
}

fn main() {
    bar_valid::<S>();
}
