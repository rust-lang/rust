//@ add-core-stubs
//@ compile-flags: --target aarch64-unknown-none-softfloat -Zmerge-functions=disabled
//@ needs-llvm-components: aarch64
#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: i64 @pass_f64_C(i64 {{[^,]*}})
#[no_mangle]
extern "C" fn pass_f64_C(x: f64) -> f64 {
    x
}

// CHECK: i64 @pass_f32_pair_C(i64 {{[^,]*}})
#[no_mangle]
extern "C" fn pass_f32_pair_C(x: (f32, f32)) -> (f32, f32) {
    x
}

// CHECK: [2 x i64] @pass_f64_pair_C([2 x i64] {{[^,]*}})
#[no_mangle]
extern "C" fn pass_f64_pair_C(x: (f64, f64)) -> (f64, f64) {
    x
}

// CHECK: i64 @pass_f64_Rust(i64 {{[^,]*}})
#[no_mangle]
fn pass_f64_Rust(x: f64) -> f64 {
    x
}

// CHECK: i64 @pass_f32_pair_Rust(i64 {{[^,]*}})
#[no_mangle]
fn pass_f32_pair_Rust(x: (f32, f32)) -> (f32, f32) {
    x
}

// CHECK: void @pass_f64_pair_Rust(ptr {{.*}}%{{[^ ]+}}, ptr {{.*}}%{{[^ ]+}})
#[no_mangle]
fn pass_f64_pair_Rust(x: (f64, f64)) -> (f64, f64) {
    x
}
