// edition:2018
#![feature(type_alias_impl_trait)]

pub trait ValidTrait {}
type ImplTrait = impl ValidTrait;

/// This returns impl trait
pub fn g() -> impl ValidTrait {
    error::_in::impl_trait()
    //~^ ERROR failed to resolve
}

/// This returns impl trait, but using a type alias
pub fn h() -> ImplTrait {
    error::_in::impl_trait::alias();
    //~^ ERROR failed to resolve
    (|| error::_in::impl_trait::alias::nested::closure())()
    //~^ ERROR failed to resolve
}

/// This used to work with ResolveBodyWithLoop.
/// However now that we ignore type checking instead of modifying the function body,
/// the return type is seen as `impl Future<Output = u32>`, not a `u32`.
/// So it no longer allows errors in the function body.
pub async fn a() -> u32 {
    error::_in::async_fn()
    //~^ ERROR failed to resolve
}
