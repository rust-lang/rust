//@ run-pass
//@ compile-flags: -Copt-level=0

// Test that floats (in particular signalling NaNs) are losslessly returned from functions.

fn main() {
    // FIXME(#114479): LLVM miscompiles loading and storing `f32` and `f64` when SSE is disabled on
    // x86.
    if cfg!(not(all(target_arch = "x86", not(target_feature = "sse2")))) {
        let bits_f32 = std::hint::black_box([
            4.2_f32.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            f32::NAN.to_bits(),
            // These two masks cover all the mantissa bits. One of them is a signalling NaN, the
            // other is quiet.
            // Similar to the masks in `test_float_bits_conv` in library/std/src/f32/tests.rs
            f32::NAN.to_bits() ^ 0x002A_AAAA,
            f32::NAN.to_bits() ^ 0x0055_5555,
            // Same as above but with the sign bit flipped.
            f32::NAN.to_bits() ^ 0x802A_AAAA,
            f32::NAN.to_bits() ^ 0x8055_5555,
        ]);
        for bits in bits_f32 {
            assert_eq!(identity(f32::from_bits(bits)).to_bits(), bits);
            // Test types that are returned as scalar pairs.
            assert_eq!(identity((f32::from_bits(bits), 42)).0.to_bits(), bits);
            assert_eq!(identity((42, f32::from_bits(bits))).1.to_bits(), bits);
            let (a, b) = identity((f32::from_bits(bits), f32::from_bits(bits)));
            assert_eq!((a.to_bits(), b.to_bits()), (bits, bits));
        }

        let bits_f64 = std::hint::black_box([
            4.2_f64.to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            f64::NAN.to_bits(),
            // These two masks cover all the mantissa bits. One of them is a signalling NaN, the
            // other is quiet.
            // Similar to the masks in `test_float_bits_conv` in library/std/src/f64/tests.rs
            f64::NAN.to_bits() ^ 0x000A_AAAA_AAAA_AAAA,
            f64::NAN.to_bits() ^ 0x0005_5555_5555_5555,
            // Same as above but with the sign bit flipped.
            f64::NAN.to_bits() ^ 0x800A_AAAA_AAAA_AAAA,
            f64::NAN.to_bits() ^ 0x8005_5555_5555_5555,
        ]);
        for bits in bits_f64 {
            assert_eq!(identity(f64::from_bits(bits)).to_bits(), bits);
            // Test types that are returned as scalar pairs.
            assert_eq!(identity((f64::from_bits(bits), 42)).0.to_bits(), bits);
            assert_eq!(identity((42, f64::from_bits(bits))).1.to_bits(), bits);
            let (a, b) = identity((f64::from_bits(bits), f64::from_bits(bits)));
            assert_eq!((a.to_bits(), b.to_bits()), (bits, bits));
        }
    }
}

#[inline(never)]
fn identity<T>(x: T) -> T {
    x
}
