// revisions: stock gated
// gate-test-const_trait_impl

#![cfg_attr(gated, feature(const_trait_impl))]
#![allow(incomplete_features)]
#![feature(rustc_attrs)]

struct S;
trait T {}
impl const T for S {}
//[stock]~^ ERROR const trait impls are experimental

#[rustc_error]
fn main() {} //[gated]~ ERROR fatal error triggered by #[rustc_error]
