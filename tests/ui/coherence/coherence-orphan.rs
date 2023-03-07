// aux-build:coherence_orphan_lib.rs
#![feature(negative_impls)]

extern crate coherence_orphan_lib as lib;

use lib::TheTrait;

struct TheType;

impl TheTrait<usize> for isize { }
//~^ ERROR E0117

impl TheTrait<TheType> for isize { }

impl TheTrait<isize> for TheType { }

impl !Send for Vec<isize> { } //~ ERROR E0117
//~^ WARNING
//~| WARNING this will change its meaning

fn main() { }
