// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck
// only-x86_64

#![feature(target_feature_11)]

#[target_feature(enable = "sse2")]
const fn sse2() {}

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
    sse2();              //~ ERROR call to function with `#[target_feature]` is unsafe
    avx_bmi2();          //~ ERROR call to function with `#[target_feature]` is unsafe
    Quux.avx_bmi2();     //~ ERROR call to function with `#[target_feature]` is unsafe
}

#[target_feature(enable = "sse2")]
fn bar() {
    avx_bmi2();          //~ ERROR call to function with `#[target_feature]` is unsafe
    Quux.avx_bmi2();     //~ ERROR call to function with `#[target_feature]` is unsafe
}

#[target_feature(enable = "avx")]
fn baz() {
    sse2();              //~ ERROR call to function with `#[target_feature]` is unsafe
    avx_bmi2();          //~ ERROR call to function with `#[target_feature]` is unsafe
    Quux.avx_bmi2();     //~ ERROR call to function with `#[target_feature]` is unsafe
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "bmi2")]
fn qux() {
    sse2();              //~ ERROR call to function with `#[target_feature]` is unsafe
}

const name: () = sse2(); //~ ERROR call to function with `#[target_feature]` is unsafe

fn main() {}
