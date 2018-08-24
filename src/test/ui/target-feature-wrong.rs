// ignore-arm
// ignore-aarch64
// ignore-wasm
// ignore-emscripten
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-s390x
// ignore-sparc
// ignore-sparc64

#![feature(target_feature)]

#[target_feature = "+sse2"]
//~^ ERROR: must be of the form
#[target_feature(enable = "foo")]
//~^ ERROR: not valid for this target
#[target_feature(bar)]
//~^ ERROR: only accepts sub-keys
#[target_feature(disable = "baz")]
//~^ ERROR: only accepts sub-keys
unsafe fn foo() {}

#[target_feature(enable = "sse2")]
//~^ ERROR: can only be applied to `unsafe` function
fn bar() {}

#[target_feature(enable = "sse2")]
//~^ ERROR: should be applied to a function
mod another {}

#[inline(always)]
//~^ ERROR: cannot use #[inline(always)]
#[target_feature(enable = "sse2")]
unsafe fn test() {}

fn main() {
    unsafe {
        foo();
        bar();
    }
}
