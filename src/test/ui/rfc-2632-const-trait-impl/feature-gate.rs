// revisions: stock gated
// gate-test-const_trait_impl

#![cfg_attr(gated, feature(const_trait_impl))]
#![allow(incomplete_features)]

struct S;
trait T {}
impl const T for S {}
//[stock]~^ ERROR const trait impls are experimental
//[stock,gated]~^^ ERROR const trait impls are not yet implemented

fn main() {}
