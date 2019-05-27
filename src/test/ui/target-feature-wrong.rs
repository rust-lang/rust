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
//~^ ERROR malformed `target_feature` attribute
#[target_feature(enable = "foo")]
//~^ ERROR not valid for this target
//~| NOTE `foo` is not valid for this target
#[target_feature(bar)]
//~^ ERROR malformed `target_feature` attribute
#[target_feature(disable = "baz")]
//~^ ERROR malformed `target_feature` attribute
unsafe fn foo() {}

#[target_feature(enable = "sse2")]
//~^ ERROR #[target_feature(..)] can only be applied to `unsafe` functions
//~| NOTE can only be applied to `unsafe` functions
fn bar() {}
//~^ NOTE not an `unsafe` function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
mod another {}
//~^ NOTE not a function

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
