// edition:2018

/// This used to work with ResolveBodyWithLoop.
/// However now that we ignore type checking instead of modifying the function body,
/// the return type is seen as `impl Future<Output = u32>`, not a `u32`.
/// So it no longer allows errors in the function body.
pub async fn a() -> u32 {
    error::_in::async_fn()
    //~^ ERROR failed to resolve
}
