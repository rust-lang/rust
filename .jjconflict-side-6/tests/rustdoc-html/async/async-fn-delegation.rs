//@ edition: 2021

// Regression test for <https://github.com/rust-lang/rust/issues/157040>.
//
// rustdoc used to ICE with "unexpected async fn return type" when cleaning a
// delegated (`reuse`) async fn: the delegation's HIR signature is unresolved
// (`InferDelegation`), so its return type cleaned to `_` even though the header
// is `async`, and unconditionally sugaring that inferred type panicked.
//
// We now clean the resolved (ty-side) signature for delegation items, like we
// already do for inlined items. That both avoids the ICE and renders the real
// return type and `self` parameter instead of `-> _` / `self: _`.
//
// Note: the `<Self>` generic on the free-function variants is a pre-existing
// quirk of how delegation generics are rendered (plain sync delegation prints it
// too); it is tracked separately and is not what this test is about.

#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![crate_name = "async_delegation"]

pub trait Trait {
    async fn unit(&self) {}
    async fn nonunit(&self) -> i32 {
        0
    }
}

//@ has async_delegation/fn.unit.html '//pre[@class="rust item-decl"]' 'pub async fn unit<Self>(&self)'
pub reuse Trait::unit;
//@ has async_delegation/fn.nonunit.html '//pre[@class="rust item-decl"]' 'pub async fn nonunit<Self>(&self) -> i32'
pub reuse Trait::nonunit;

pub struct S;
impl Trait for S {}

//@ has async_delegation/struct.S.html '//*[@class="code-header"]' 'pub async fn unit(self: &S)'
//@ has async_delegation/struct.S.html '//*[@class="code-header"]' 'pub async fn nonunit(self: &S) -> i32'
impl S {
    pub reuse Trait::unit { self }
    pub reuse Trait::nonunit { self }
}
