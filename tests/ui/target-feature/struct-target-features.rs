//@ only-x86_64
#![feature(struct_target_features)]
//~^ WARNING the feature `struct_target_features` is incomplete and may not be safe to use and/or cause compiler crashes
#![feature(target_feature_11)]

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
struct Invalid(u32);

#[target_feature(enable = "avx")]
struct Avx {}

#[target_feature(enable = "sse")]
struct Sse();

#[target_feature(enable = "avx")]
fn avx() {}

trait TFAssociatedType {
    type Assoc;
}

impl TFAssociatedType for () {
    type Assoc = Avx;
}

#[target_feature(from_args)]
fn avx_self(_: <() as TFAssociatedType>::Assoc) {
    avx();
}

#[target_feature(from_args)]
fn avx_avx(_: Avx) {
    avx();
}

#[target_feature(from_args)]
extern "C" fn bad_fun(_: Avx) {}

#[inline(always)]
//~^ ERROR cannot use `#[inline(always)]` with `#[target_feature]`
#[target_feature(from_args)]
fn inline_fun(_: Avx) {}
//~^ ERROR cannot use a struct with target features in a #[inline(always)] function

trait Simd {
    fn do_something(&self);
}

impl Simd for Avx {
    #[target_feature(from_args)]
    fn do_something(&self) {
        unsafe {
            println!("{:?}", _mm256_setzero_ps());
        }
    }
}

impl Simd for Sse {
    #[target_feature(from_args)]
    fn do_something(&self) {
        unsafe {
            println!("{:?}", _mm_setzero_ps());
        }
    }
}

#[target_feature(from_args)]
fn avxfn(_: &Avx) {
    // This is not unsafe because we already have the feature at function-level.
    let _ = Avx {};
}

fn main() {
    Avx {};
    //~^ ERROR initializing type `Avx` with `#[target_feature]` is unsafe and requires unsafe function or block [E0133]

    if is_x86_feature_detected!("avx") {
        let avx = unsafe { Avx {} };
        avxfn(&avx);
        avx.do_something();
    }
    if is_x86_feature_detected!("sse") {
        unsafe { Sse {} }.do_something();
    }
}
