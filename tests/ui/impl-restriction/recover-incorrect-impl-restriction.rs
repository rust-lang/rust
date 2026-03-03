//@ compile-flags: --crate-type=lib
//@ revisions: with_gate without_gate
#![cfg_attr(with_gate, feature(impl_restriction))]
//[with_gate]~^ WARN the feature `impl_restriction` is incomplete and may not be safe to use and/or cause compiler crashes
#![feature(auto_traits, const_trait_impl)]

mod foo {
    pub impl(crate::foo) trait Baz {} //~ ERROR incorrect `impl` restriction
    //[without_gate]~^ ERROR `impl` restrictions are experimental
    pub unsafe impl(crate::foo) trait BazUnsafe {} //~ ERROR incorrect `impl` restriction
    //[without_gate]~^ ERROR `impl` restrictions are experimental
    pub auto impl(crate::foo) trait BazAuto {} //~ ERROR incorrect `impl` restriction
    //[without_gate]~^ ERROR `impl` restrictions are experimental
    pub const impl(crate::foo) trait BazConst {} //~ ERROR incorrect `impl` restriction
    //[without_gate]~^ ERROR `impl` restrictions are experimental
    pub const unsafe impl(crate::foo) trait BazConstUnsafe {} //~ ERROR incorrect `impl` restriction
    //[without_gate]~^ ERROR `impl` restrictions are experimental
    pub unsafe auto impl(crate::foo) trait BazUnsafeAuto {} //~ ERROR incorrect `impl` restriction
    //[without_gate]~^ ERROR `impl` restrictions are experimental
}
