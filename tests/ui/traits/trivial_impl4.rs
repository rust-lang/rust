//! Check that we don't break orphan rules.
//! The dependency may add an impl for `u8` later,
//! which would break this crate. We want to avoid adding
//! more ways in which adding an impl can be a breaking change.
//! This test differs from `trivial_impl3` because there actually
//! exists any impl for `Trait`, which has an effect on coherence.

//@ aux-build:trivial4.rs

extern crate trivial4;

pub trait Foo {
    fn foo()
    where
        Self: trivial4::Trait;
}

impl Foo for u8 {}
//~^ ERROR not all trait items implemented, missing: `foo`

fn main() {}
