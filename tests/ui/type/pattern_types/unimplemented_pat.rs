//! This test ensures we do not ICE for unimplemented
//! patterns unless the feature gate is enabled.

#![feature(pattern_type_macro)]

use std::pat::pattern_type;

type Always = pattern_type!(Option<u32> is Some(_));
//~^ ERROR: pattern types are unstable

type Binding = pattern_type!(Option<u32> is x);
//~^ ERROR: pattern types are unstable

fn main() {}
