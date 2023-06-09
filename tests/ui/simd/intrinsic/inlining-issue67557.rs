// This used to cause assert_10_13 to unexpectingly fail, due to simd_shuffle_indices being passed
// the wrong Instance, causing issues with inlining. See #67557.
//
// run-pass
// compile-flags: -Zmir-opt-level=4
#![feature(platform_intrinsics, repr_simd)]

extern "platform-intrinsic" {
    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
}

#[repr(simd)]
#[derive(Debug, PartialEq)]
struct Simd2(u8, u8);

fn main() {
    unsafe {
        const IDX: [u32; 2] = [0, 1];
        let p_res: Simd2 = simd_shuffle2(Simd2(10, 11), Simd2(12, 13), IDX);
        let a_res: Simd2 = inline_me();

        assert_10_11(p_res);
        assert_10_13(a_res);
    }
}

#[inline(never)]
fn assert_10_11(x: Simd2) {
    assert_eq!(x, Simd2(10, 11));
}

#[inline(never)]
fn assert_10_13(x: Simd2) {
    assert_eq!(x, Simd2(10, 13));
}


#[inline(always)]
unsafe fn inline_me() -> Simd2 {
    const IDX: [u32; 2] = [0, 3];
    simd_shuffle2(Simd2(10, 11), Simd2(12, 13), IDX)
}
