//@ only-x86_64

#![feature(target_feature_11)]

#[target_feature(enable = "sse2")]
const fn sse2() {}

#[target_feature(enable = "sse2")]
#[target_feature(enable = "fxsr")]
const fn sse2_and_fxsr() {}

#[target_feature(enable = "avx")]
#[target_feature(enable = "bmi2")]
fn avx_bmi2() {}

struct Quux;

impl Quux {
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "bmi2")]
    fn avx_bmi2(&self) {}
}

fn foo() {
    sse2();
    //~^ ERROR call to function `sse2` with `#[target_feature]` is unsafe
    avx_bmi2();
    //~^ ERROR call to function `avx_bmi2` with `#[target_feature]` is unsafe
    Quux.avx_bmi2();
    //~^ ERROR call to function `Quux::avx_bmi2` with `#[target_feature]` is unsafe
}

#[target_feature(enable = "sse2")]
fn bar() {
    avx_bmi2();
    //~^ ERROR call to function `avx_bmi2` with `#[target_feature]` is unsafe
    Quux.avx_bmi2();
    //~^ ERROR call to function `Quux::avx_bmi2` with `#[target_feature]` is unsafe
}

#[target_feature(enable = "avx")]
fn baz() {
    sse2();
    //~^ ERROR call to function `sse2` with `#[target_feature]` is unsafe
    avx_bmi2();
    //~^ ERROR call to function `avx_bmi2` with `#[target_feature]` is unsafe
    Quux.avx_bmi2();
    //~^ ERROR call to function `Quux::avx_bmi2` with `#[target_feature]` is unsafe
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "bmi2")]
fn qux() {
    sse2();
    //~^ ERROR call to function `sse2` with `#[target_feature]` is unsafe
}

const _: () = sse2();
//~^ ERROR call to function `sse2` with `#[target_feature]` is unsafe

const _: () = sse2_and_fxsr();
//~^ ERROR call to function `sse2_and_fxsr` with `#[target_feature]` is unsafe

#[deny(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
#[target_feature(enable = "bmi2")]
unsafe fn needs_unsafe_block() {
    sse2();
    //~^ ERROR call to function `sse2` with `#[target_feature]` is unsafe
}

fn main() {}
