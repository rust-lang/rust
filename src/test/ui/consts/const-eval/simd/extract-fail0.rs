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
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

const X: i8x1 = i8x1(42);

const fn extract_wrong_ret() -> i16 {
    unsafe { simd_extract(X, 0_u32) }
}

const A: i16 = extract_wrong_ret();

fn main() {}
