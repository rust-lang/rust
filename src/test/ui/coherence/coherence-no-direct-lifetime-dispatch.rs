// Test that you cannot *directly* dispatch on lifetime requirements

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

trait MyTrait { fn foo() {} }

impl<T> MyTrait for T {}
impl<T: 'static> MyTrait for T {}
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119

fn main() {}
