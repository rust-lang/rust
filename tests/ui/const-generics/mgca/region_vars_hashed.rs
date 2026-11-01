//! Regression test for https://github.com/rust-lang/rust/issues/157147.
//!
//! It used to ICE when trying to process region vars in const arguments,
//! Since it tries to hash intentionally un-hashable `ReVar` inference var.

//@ compile-flags: --crate-type=lib
//@ incremental
#![feature(min_generic_const_args)]

fn foo<const N: usize>() {
    foo::<{ Some::<&u8> { 0: const { &0_u8 } } }>()
    //~^ ERROR: anonymous constants with lifetimes in their type are not yet supported
}
