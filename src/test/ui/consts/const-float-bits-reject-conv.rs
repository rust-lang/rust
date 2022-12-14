// compile-flags: -Zmir-opt-level=0
// error-pattern: cannot use f32::to_bits on a NaN
#![feature(const_float_bits_conv)]
#![feature(const_float_classify)]

// Don't promote
const fn nop<T>(x: T) -> T { x }

macro_rules! const_assert {
    ($a:expr) => {
        {
            const _: () = assert!($a);
            assert!(nop($a));
        }
    };
    ($a:expr, $b:expr) => {
        {
            const _: () = assert!($a == $b);
            assert_eq!(nop($a), nop($b));
        }
    };
}

fn f32() {
    // Check that NaNs roundtrip their bits regardless of signalingness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    // ...actually, let's just check that these break. :D
    const MASKED_NAN1: u32 = f32::NAN.to_bits() ^ 0x002A_AAAA;
    //~^ inside
    const MASKED_NAN2: u32 = f32::NAN.to_bits() ^ 0x0055_5555;
    //~^ inside

    // The rest of the code is dead because the constants already fail to evaluate.

    const_assert!(f32::from_bits(MASKED_NAN1).is_nan());
    const_assert!(f32::from_bits(MASKED_NAN1).is_nan());

    // LLVM does not guarantee that loads and stores of NaNs preserve their exact bit pattern.
    // In practice, this seems to only cause a problem on x86, since the most widely used calling
    // convention mandates that floating point values are returned on the x87 FPU stack. See #73328.
    // However, during CTFE we still preserve bit patterns (though that is not a guarantee).
    const_assert!(f32::from_bits(MASKED_NAN1).to_bits(), MASKED_NAN1);
    const_assert!(f32::from_bits(MASKED_NAN2).to_bits(), MASKED_NAN2);
}

fn f64() {
    // Check that NaNs roundtrip their bits regardless of signalingness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    // ...actually, let's just check that these break. :D
    const MASKED_NAN1: u64 = f64::NAN.to_bits() ^ 0x000A_AAAA_AAAA_AAAA;
    //~^ inside
    const MASKED_NAN2: u64 = f64::NAN.to_bits() ^ 0x0005_5555_5555_5555;
    //~^ inside

    // The rest of the code is dead because the constants already fail to evaluate.

    const_assert!(f64::from_bits(MASKED_NAN1).is_nan());
    const_assert!(f64::from_bits(MASKED_NAN1).is_nan());

    // See comment above.
    const_assert!(f64::from_bits(MASKED_NAN1).to_bits(), MASKED_NAN1);
    const_assert!(f64::from_bits(MASKED_NAN2).to_bits(), MASKED_NAN2);
}

fn main() {
    f32();
    f64();
}
