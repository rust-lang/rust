// run-pass
// ignore-emscripten

// Test that the simd_f{min,max} intrinsics produce the correct results.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct f32x4(pub f32, pub f32, pub f32, pub f32);

extern "platform-intrinsic" {
    fn simd_fmin<T>(x: T, y: T) -> T;
    fn simd_fmax<T>(x: T, y: T) -> T;
}

fn main() {
    let x = f32x4(1.0, 2.0, 3.0, 4.0);
    let y = f32x4(2.0, 1.0, 4.0, 3.0);

    #[cfg(not(any(target_arch = "mips", target_arch = "mips64")))]
    let nan = f32::NAN;
    // MIPS hardware treats f32::NAN as SNAN. Clear the signaling bit.
    // See https://github.com/rust-lang/rust/issues/52746.
    #[cfg(any(target_arch = "mips", target_arch = "mips64"))]
    let nan = f32::from_bits(f32::NAN.to_bits() - 1);

    let n = f32x4(nan, nan, nan, nan);

    unsafe {
        let min0 = simd_fmin(x, y);
        let min1 = simd_fmin(y, x);
        assert_eq!(min0, min1);
        let e = f32x4(1.0, 1.0, 3.0, 3.0);
        assert_eq!(min0, e);
        let minn = simd_fmin(x, n);
        assert_eq!(minn, x);
        let minn = simd_fmin(y, n);
        assert_eq!(minn, y);

        let max0 = simd_fmax(x, y);
        let max1 = simd_fmax(y, x);
        assert_eq!(max0, max1);
        let e = f32x4(2.0, 2.0, 4.0, 4.0);
        assert_eq!(max0, e);
        let maxn = simd_fmax(x, n);
        assert_eq!(maxn, x);
        let maxn = simd_fmax(y, n);
        assert_eq!(maxn, y);
    }
}
