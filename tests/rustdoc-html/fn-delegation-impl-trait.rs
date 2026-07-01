// Regression test for <https://github.com/rust-lang/rust/issues/155728>
// Make sure delegating functions with `impl Trait` in argument position works correctly.
//@ aux-crate:aux=fn-delegation-impl-trait-aux.rs
//@ edition: 2021
#![feature(fn_delegation)]
#![allow(incomplete_features)]

fn foo(_: impl FnOnce()) {}

//@ has fn_delegation_impl_trait/fn.top_level.html '//pre[@class="rust item-decl"]' 'pub fn top_level(arg0: impl FnOnce())'
pub reuse foo as top_level;

pub struct S;

//@ has fn_delegation_impl_trait/struct.S.html '//*[@id="method.method"]' 'pub fn method(arg0: impl FnOnce())'
impl S {
    pub reuse foo as method;
}

pub trait A {
    fn f(&self, _: impl FnOnce()) {}
}

impl A for S {}

//@ has fn_delegation_impl_trait/struct.S.html '//*[@id="method.f"]' 'pub fn f(self: &S, arg1: impl FnOnce())'
//@ !has fn_delegation_impl_trait/struct.S.html '//*[@id="method.f"]' 'MetaSized'
impl S {
    pub reuse <S as A>::f;
}

//@ has fn_delegation_impl_trait/trait.T.html '//*[@id="method.provided"]' 'fn provided(arg0: impl FnOnce())'
pub trait T {
    reuse foo as provided;
}

//@ has fn_delegation_impl_trait/fn.cross_crate.html '//pre[@class="rust item-decl"]' 'pub fn cross_crate(arg0: impl FnOnce())'
pub reuse aux::external as cross_crate;

//@ has fn_delegation_impl_trait/fn.inlined_cross_crate_delegated.html '//pre[@class="rust item-decl"]' 'pub fn inlined_cross_crate_delegated(arg0: impl FnOnce())'
#[doc(inline)]
pub use aux::delegated as inlined_cross_crate_delegated;

//@ has fn_delegation_impl_trait/fn.redelegated_cross_crate.html '//pre[@class="rust item-decl"]' 'pub fn redelegated_cross_crate(arg0: impl FnOnce())'
pub reuse aux::delegated as redelegated_cross_crate;

pub fn main() {}
