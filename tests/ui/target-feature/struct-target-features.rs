//@ only-x86_64
#![feature(struct_target_features)]
//~^ WARNING the feature `struct_target_features` is incomplete and may not be safe to use and/or cause compiler crashes
#![feature(target_feature_11)]

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
//~^ ERROR attribute should be applied to a function definition or unit struct
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

fn avx_self(_: <() as TFAssociatedType>::Assoc) {
    avx();
}

fn avx_avx(_: Avx) {
    avx();
}

extern "C" fn bad_fun(_: Avx) {}
//~^ ERROR cannot use a struct with target features in a function with non-Rust ABI

#[inline(always)]
//~^ ERROR cannot use `#[inline(always)]` with `#[target_feature]`
fn inline_fun(_: Avx) {}
//~^ ERROR cannot use a struct with target features in a #[inline(always)] function

trait Simd {
    fn do_something(&self);
}

impl Simd for Avx {
    fn do_something(&self) {
        unsafe {
            println!("{:?}", _mm256_setzero_ps());
        }
    }
}

impl Simd for Sse {
    fn do_something(&self) {
        unsafe {
            println!("{:?}", _mm_setzero_ps());
        }
    }
}

struct WithAvx {
    #[allow(dead_code)]
    avx: Avx,
}

impl Simd for WithAvx {
    fn do_something(&self) {
        unsafe {
            println!("{:?}", _mm256_setzero_ps());
        }
    }
}

#[inline(never)]
fn dosomething<S: Simd>(simd: &S) {
    simd.do_something();
}

fn avxfn(_: &Avx) {}

fn main() {
    Avx {};
    //~^ ERROR initializing type with `target_feature` attr is unsafe and requires unsafe function or block [E0133]

    if is_x86_feature_detected!("avx") {
        let avx = unsafe { Avx {} };
        avxfn(&avx);
        dosomething(&avx);
        dosomething(&WithAvx { avx });
    }
    if is_x86_feature_detected!("sse") {
        dosomething(&unsafe { Sse {} })
    }
}
