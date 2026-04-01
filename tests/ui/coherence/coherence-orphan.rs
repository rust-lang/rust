//@ aux-build:coherence_orphan_lib.rs
#![feature(negative_impls)]

extern crate coherence_orphan_lib as lib;

use lib::TheTrait;

struct TheType;

impl TheTrait<usize> for isize {}
//~^ ERROR  only traits defined in the current crate can be implemented for primitive types

impl TheTrait<TheType> for isize {}

impl TheTrait<isize> for TheType {}

impl !Send for Vec<isize> {}
//~^ ERROR only traits defined in the current crate can be implemented for types defined outside of the crate

fn main() {}
