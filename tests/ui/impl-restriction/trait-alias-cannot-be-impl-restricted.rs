//@ compile-flags: --crate-type=lib
//@ revisions: with_gate without_gate
#![cfg_attr(with_gate, feature(impl_restriction))]
//[with_gate]~^ WARN the feature `impl_restriction` is incomplete and may not be safe to use and/or cause compiler crashes
#![feature(auto_traits, const_trait_impl, trait_alias)]

impl(crate) trait Alias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
//[without_gate]~^ ERROR `impl` restrictions are experimental
auto impl(in crate) trait AutoAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
//~^ ERROR trait aliases cannot be `auto`
//[without_gate]~| ERROR `impl` restrictions are experimental
unsafe impl(self) trait UnsafeAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
//~^ ERROR trait aliases cannot be `unsafe`
//[without_gate]~| ERROR `impl` restrictions are experimental
const impl(in self) trait ConstAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
//[without_gate]~^ ERROR `impl` restrictions are experimental

mod foo {
    impl(super) trait InnerAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
    //[without_gate]~^ ERROR `impl` restrictions are experimental
    const unsafe impl(in crate::foo) trait InnerConstUnsafeAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
    //~^ ERROR trait aliases cannot be `unsafe`
    //[without_gate]~| ERROR `impl` restrictions are experimental
    unsafe auto impl(in crate::foo) trait InnerUnsafeAutoAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
    //~^ ERROR trait aliases cannot be `auto`
    //~^^ ERROR trait aliases cannot be `unsafe`
    //[without_gate]~| ERROR `impl` restrictions are experimental
}
