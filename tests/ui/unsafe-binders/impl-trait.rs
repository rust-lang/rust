//@ compile-flags: -Znext-solver
//@ edition: 2024

// Opaque types cannot capture lifetiems from unsafe binders, but can be in their
// hidden types.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::wrap_binder;

fn in_binder_return() -> unsafe<'a> impl Sized {
    //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime
    todo!()
}

fn captures_bound(_: unsafe<'a> &'a impl Sized) {}

fn in_binder_arg(_: unsafe<'a> impl Sized) {}
//~^ ERROR the trait bound `impl Sized: Copy` is not satisfied

fn binder_through_opaque() -> impl Sized {
    let b: unsafe<'a> &'a u8 = unsafe { wrap_binder!(&0) };
    b
}

fn main() {}
