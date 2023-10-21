// Check that the layout of a coroutine is available when auxiliary crate
// is compiled with --emit metadata.
//
// Regression test for #80998.
//
// aux-build:metadata-sufficient-for-layout.rs

#![feature(type_alias_impl_trait, rustc_attrs)]
#![feature(coroutine_trait)]

extern crate metadata_sufficient_for_layout;

use std::ops::Coroutine;

type F = impl Coroutine<(), Yield = (), Return = ()>;

// Static queries the layout of the coroutine.
static A: Option<F> = None;

fn f() -> F {
    metadata_sufficient_for_layout::g()
}

#[rustc_error]
fn main() {} //~ ERROR
