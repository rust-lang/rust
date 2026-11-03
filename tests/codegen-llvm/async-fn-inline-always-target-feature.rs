// `#[inline(always)]` on an `async fn` applies to its body coroutine (#129347). Unlike a closure
// marked `#[inline(always)]`, that coroutine is never inlined into the function it was written
// in, so it must still inherit that function's `#[target_feature]`s.
//
//@ revisions: x86_64 aarch64
//@[x86_64] only-x86_64
//@[aarch64] only-aarch64
//@ edition: 2024
//@ compile-flags: -Cno-prepopulate-passes -Csymbol-mangling-version=v0 -Zinline-mir=no
#![crate_type = "lib"]

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Waker};

// CHECK:      ; async_fn_inline_always_target_feature::always::{closure#0}
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define {{.*}} #[[ATTRS:[0-9]+]]
#[inline(always)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "sha3"))]
pub async fn always() {}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "sha3"))]
pub fn poll_always() {
    let mut cx = Context::from_waker(Waker::noop());
    let _ = pin!(always()).poll(&mut cx);
}

// x86_64:  attributes #[[ATTRS]] = {{{.*}}"target-features"="{{[^"]*}}+avx2
// aarch64: attributes #[[ATTRS]] = {{{.*}}"target-features"="{{[^"]*}}+sha3
