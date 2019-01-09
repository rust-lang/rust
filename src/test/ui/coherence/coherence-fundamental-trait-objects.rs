// Check that trait objects from #[fundamental] traits are not
// treated as #[fundamental] types - the 2 meanings of #[fundamental]
// are distinct.

// aux-build:coherence_fundamental_trait_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_fundamental_trait_lib;

use coherence_fundamental_trait_lib::{Fundamental, Misc};

pub struct Local;
impl Misc for dyn Fundamental<Local> {}
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

fn main() {}
