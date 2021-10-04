// compile-flags: -Zmir-opt-level=0
// run-pass

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
    const_assert!((1f32).to_bits(), 0x3f800000);
    const_assert!(u32::from_be_bytes(1f32.to_be_bytes()), 0x3f800000);
    const_assert!((12.5f32).to_bits(), 0x41480000);
    const_assert!(u32::from_le_bytes(12.5f32.to_le_bytes()), 0x41480000);
    const_assert!((1337f32).to_bits(), 0x44a72000);
    const_assert!(u32::from_ne_bytes(1337f32.to_ne_bytes()), 0x44a72000);
    const_assert!((-14.25f32).to_bits(), 0xc1640000);
    const_assert!(f32::from_bits(0x3f800000), 1.0);
    const_assert!(f32::from_be_bytes(0x3f800000u32.to_be_bytes()), 1.0);
    const_assert!(f32::from_bits(0x41480000), 12.5);
    const_assert!(f32::from_le_bytes(0x41480000u32.to_le_bytes()), 12.5);
    const_assert!(f32::from_bits(0x44a72000), 1337.0);
    const_assert!(f32::from_ne_bytes(0x44a72000u32.to_ne_bytes()), 1337.0);
    const_assert!(f32::from_bits(0xc1640000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signalingness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    const MASKED_NAN1: u32 = f32::NAN.to_bits() ^ 0x002A_AAAA;
    const MASKED_NAN2: u32 = f32::NAN.to_bits() ^ 0x0055_5555;

    const_assert!(f32::from_bits(MASKED_NAN1).is_nan());
    const_assert!(f32::from_bits(MASKED_NAN1).is_nan());

    // LLVM does not guarantee that loads and stores of NaNs preserve their exact bit pattern.
    // In practice, this seems to only cause a problem on x86, since the most widely used calling
    // convention mandates that floating point values are returned on the x87 FPU stack. See #73328.
    if !cfg!(target_arch = "x86") {
        const_assert!(f32::from_bits(MASKED_NAN1).to_bits(), MASKED_NAN1);
        const_assert!(f32::from_bits(MASKED_NAN2).to_bits(), MASKED_NAN2);
    }
}

fn f64() {
    const_assert!((1f64).to_bits(), 0x3ff0000000000000);
    const_assert!(u64::from_be_bytes(1f64.to_be_bytes()), 0x3ff0000000000000);
    const_assert!((12.5f64).to_bits(), 0x4029000000000000);
    const_assert!(u64::from_le_bytes(12.5f64.to_le_bytes()), 0x4029000000000000);
    const_assert!((1337f64).to_bits(), 0x4094e40000000000);
    const_assert!(u64::from_ne_bytes(1337f64.to_ne_bytes()), 0x4094e40000000000);
    const_assert!((-14.25f64).to_bits(), 0xc02c800000000000);
    const_assert!(f64::from_bits(0x3ff0000000000000), 1.0);
    const_assert!(f64::from_be_bytes(0x3ff0000000000000u64.to_be_bytes()), 1.0);
    const_assert!(f64::from_bits(0x4029000000000000), 12.5);
    const_assert!(f64::from_le_bytes(0x4029000000000000u64.to_le_bytes()), 12.5);
    const_assert!(f64::from_bits(0x4094e40000000000), 1337.0);
    const_assert!(f64::from_ne_bytes(0x4094e40000000000u64.to_ne_bytes()), 1337.0);
    const_assert!(f64::from_bits(0xc02c800000000000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signalingness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    const MASKED_NAN1: u64 = f64::NAN.to_bits() ^ 0x000A_AAAA_AAAA_AAAA;
    const MASKED_NAN2: u64 = f64::NAN.to_bits() ^ 0x0005_5555_5555_5555;

    const_assert!(f64::from_bits(MASKED_NAN1).is_nan());
    const_assert!(f64::from_bits(MASKED_NAN1).is_nan());

    // See comment above.
    if !cfg!(target_arch = "x86") {
        const_assert!(f64::from_bits(MASKED_NAN1).to_bits(), MASKED_NAN1);
        const_assert!(f64::from_bits(MASKED_NAN2).to_bits(), MASKED_NAN2);
    }
}

fn main() {
    f32();
    f64();
}
