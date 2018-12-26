// run-pass
// ignore-emscripten FIXME(#45351)

#![feature(platform_intrinsics, repr_simd)]

extern "platform-intrinsic" {
    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
}

#[repr(simd)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct u64x2(u64, u64);

fn main() {
    let a = u64x2(1, 2);
    let r: u64x2 = unsafe { simd_shuffle2(a, a, [0-0, 0-0]) };
    assert_eq!(r.0, 1);
    assert_eq!(r.1, 1);
}
