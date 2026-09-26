// Inlining attributes on an `async fn` apply to its body coroutine instead of the function
// constructing it (#129347). The body coroutine inherits the function's target features, so
// `#[inline(always)]` must still be rejected together with `#[target_feature]`.
//
//@ revisions: x86_64 aarch64
//@[x86_64] only-x86_64
//@[aarch64] only-aarch64
//@ edition: 2024

#![crate_type = "lib"]
#![feature(stmt_expr_attributes)]

#[inline(always)]
//~^ ERROR cannot use `#[inline(always)]` with `#[target_feature]`
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "sse2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub async fn always() {}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "sse2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub async fn never() {}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "sse2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub fn with_async_closure() {
    // Just like a regular `#[inline(always)]` closure, this doesn't inherit the target features.
    let _ = #[inline(always)] async || {};
}
