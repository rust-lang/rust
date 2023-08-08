// run-pass
// compile-flags: -O -Zverify-llvm-ir

#![feature(repr_simd)]
#![feature(platform_intrinsics)]

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(simd)]
struct i32x4([i32; 4]);

extern "platform-intrinsic" {
    pub(crate) fn simd_add<T>(x: T, y: T) -> T;
}

#[inline(always)]
fn to_array(a: i32x4) -> [i32; 4] {
    a.0
}

fn main() {
    let a = i32x4([1, 2, 3, 4]);
    let b = unsafe { simd_add(a, a) };
    assert_eq!(to_array(b), [2, 4, 6, 8]);
}
