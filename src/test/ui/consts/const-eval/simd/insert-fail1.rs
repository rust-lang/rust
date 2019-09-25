// failure-status: 101
// rustc-env:RUST_BACKTRACE=0
// normalize-stderr-test "note: rustc 1.* running on .*" -> "note: rustc VERSION running on TARGET"
// normalize-stderr-test "note: compiler flags: .*" -> "note: compiler flags: FLAGS"
// normalize-stderr-test "interpret/intern.rs:[0-9]*:[0-9]*" -> "interpret/intern.rs:LL:CC"

#![feature(const_fn)]
#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)] struct i8x1(i8);

extern "platform-intrinsic" {
    fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
}

const X: i8x1 = i8x1(42);

const fn insert_wrong_idx() -> i8x1 {
    unsafe { simd_insert(X, 1_u32, 42_i8) }
}

const E: i8x1 = insert_wrong_idx();

fn main() {}
