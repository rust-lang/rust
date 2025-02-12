//! This test ensures we do not ICE for unimplemented
//! patterns even if the feature gate is enabled.

#![feature(pattern_type_macro, pattern_types)]

use std::pat::pattern_type;

type Always = pattern_type!(Option<u32> is Some(_));
//~^ ERROR: pattern not supported

type Binding = pattern_type!(Option<u32> is x);
//~^ ERROR: pattern not supported

fn main() {}
