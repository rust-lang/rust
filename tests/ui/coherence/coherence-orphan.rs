//@ aux-build:coherence_orphan_lib.rs
#![feature(negative_impls)]

extern crate coherence_orphan_lib as lib;

use lib::TheTrait;

struct TheType;

impl TheTrait<usize> for isize {}
//~^ ERROR E0117
//~| ERROR not all trait items implemented

impl TheTrait<TheType> for isize {}
//~^ ERROR not all trait items implemented

impl TheTrait<isize> for TheType {}
//~^ ERROR not all trait items implemented

impl !Send for Vec<isize> {} //~ ERROR E0117

fn main() {}
