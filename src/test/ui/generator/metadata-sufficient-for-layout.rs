// Check that the layout of a generator is available when auxiliary crate
// is compiled with --emit metadata.
//
// Regression test for #80998.
//
// aux-build:metadata-sufficient-for-layout.rs

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait, rustc_attrs)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![feature(generator_trait)]

extern crate metadata_sufficient_for_layout;

use std::ops::Generator;

type F = impl Generator<(), Yield = (), Return = ()>;

// Static queries the layout of the generator.
static A: Option<F> = None;

fn f() -> F {
    metadata_sufficient_for_layout::g()
}

#[rustc_error]
fn main() {} //~ ERROR
