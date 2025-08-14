//@ revisions: stock gated
//@[gated] check-pass
// gate-test-const_trait_impl

#![cfg_attr(gated, feature(const_trait_impl))]

struct S;
#[const_trait] //[stock]~ ERROR `const_trait` is a temporary placeholder
trait T {}
impl const T for S {}
//[stock]~^ ERROR const trait impls are experimental

const fn f<A: [const] T>() {} //[stock]~ ERROR const trait impls are experimental
fn g<A: const T>() {} //[stock]~ ERROR const trait impls are experimental

const trait Trait {} //[stock]~ ERROR const trait impls are experimental
#[cfg(false)] const trait Trait {} //[stock]~ ERROR const trait impls are experimental

macro_rules! discard { ($ty:ty) => {} }

discard! { impl [const] T } //[stock]~ ERROR const trait impls are experimental
discard! { impl const T } //[stock]~ ERROR const trait impls are experimental

fn main() {}
