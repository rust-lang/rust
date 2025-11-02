//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(pattern_types, generic_pattern_types, pattern_type_macro)]
#![expect(incomplete_features)]

use std::pat::pattern_type;

trait Foo {}

impl<const START: u32, const END: u32> Foo for pattern_type!(u32 is START..=END) {}

fn main() {}
