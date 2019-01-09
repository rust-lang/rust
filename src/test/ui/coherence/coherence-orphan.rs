// aux-build:coherence_orphan_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]

extern crate coherence_orphan_lib as lib;

use lib::TheTrait;

struct TheType;

impl TheTrait<usize> for isize { }
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

impl TheTrait<TheType> for isize { }

impl TheTrait<isize> for TheType { }

impl !Send for Vec<isize> { }
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

fn main() { }
