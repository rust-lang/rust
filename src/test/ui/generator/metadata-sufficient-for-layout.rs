// Check that the layout of a generator is available when auxiliary crate
// is compiled with --emit metadata.
//
// Regression test for #80998.
//
// aux-build:metadata-sufficient-for-layout.rs
// check-pass

#![feature(type_alias_impl_trait)]
#![feature(generator_trait)]

extern crate metadata_sufficient_for_layout;

use std::ops::Generator;

type F = impl Generator<(), Yield = (), Return = ()>;

// Static queries the layout of the generator.
static A: Option<F> = None;

fn f() -> F { metadata_sufficient_for_layout::g() }

fn main() {}
