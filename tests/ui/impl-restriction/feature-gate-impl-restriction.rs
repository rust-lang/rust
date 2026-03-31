//@ compile-flags: --crate-type=lib
//@ revisions: with_gate without_gate
//@[with_gate] check-pass

#![cfg_attr(with_gate, feature(impl_restriction))]
#![cfg_attr(with_gate, allow(incomplete_features))]
#![feature(auto_traits, const_trait_impl)]

pub impl(crate) trait Bar {} //[without_gate]~ ERROR `impl` restrictions are experimental
pub impl(in crate) trait BarInCrate {} //[without_gate]~ ERROR `impl` restrictions are experimental

mod foo {
    pub impl(in crate::foo) trait Baz {} //[without_gate]~ ERROR `impl` restrictions are experimental
    pub unsafe impl(super) trait BazUnsafeSuper {} //[without_gate]~ ERROR `impl` restrictions are experimental
    pub auto impl(self) trait BazAutoSelf {} //[without_gate]~ ERROR `impl` restrictions are experimental
    pub const impl(in self) trait BazConst {} //[without_gate]~ ERROR `impl` restrictions are experimental

    mod foo_inner {
        pub impl(in crate::foo::foo_inner) trait Qux {} //[without_gate]~ ERROR `impl` restrictions are experimental
        pub unsafe auto impl(in crate::foo::foo_inner) trait QuxAutoUnsafe {} //[without_gate]~ ERROR `impl` restrictions are experimental
        pub const unsafe impl(in crate::foo::foo_inner) trait QuxConstUnsafe {} //[without_gate]~ ERROR `impl` restrictions are experimental
    }

    #[cfg(false)]
    pub impl(crate) trait Bar {} //[without_gate]~ ERROR `impl` restrictions are experimental
    #[cfg(false)]
    pub impl(in crate) trait BarInCrate {} //[without_gate]~ ERROR `impl` restrictions are experimental
    #[cfg(false)]
    pub unsafe impl(self) trait BazUnsafeSelf {} //[without_gate]~ ERROR `impl` restrictions are experimental
    #[cfg(false)]
    pub auto impl(in super) trait BazAutoSuper {} //[without_gate]~ ERROR `impl` restrictions are experimental
    #[cfg(false)]
    pub const impl(super) trait BazConstSuper {} //[without_gate]~ ERROR `impl` restrictions are experimental

    #[cfg(false)]
    mod cfged_out_foo {
        pub impl(in crate::foo::cfged_out_foo) trait CfgedOutQux {} //[without_gate]~ ERROR `impl` restrictions are experimental
        pub unsafe auto impl(in crate::foo::cfged_out_foo) trait CfgedOutQuxUnsafeAuto {} //[without_gate]~ ERROR `impl` restrictions are experimental
        pub const unsafe impl(in crate::foo::cfged_out_foo) trait CfgedOutQuxConstUnsafe {} //[without_gate]~ ERROR `impl` restrictions are experimental
    }

    // auto traits cannot be const, so we do not include these combinations in the test.
}
