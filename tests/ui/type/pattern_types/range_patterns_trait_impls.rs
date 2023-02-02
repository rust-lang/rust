#![feature(pattern_types, rustc_attrs)]
#![feature(core_pattern_type)]
#![feature(core_pattern_types)]
#![allow(incomplete_features)]

//! check that pattern types can have local traits
//! implemented for them.

//@ check-pass

use std::pat::pattern_type;

type Y = pattern_type!(u32 is 1..);

trait Trait {}

impl Trait for Y {}

fn main() {}
