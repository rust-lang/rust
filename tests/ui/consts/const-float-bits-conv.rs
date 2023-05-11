// compile-flags: -Zmir-opt-level=0
// run-pass

#![feature(const_float_bits_conv)]
#![feature(const_float_classify)]
#![allow(unused_macro_rules)]

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
}

fn main() {
    f32();
    f64();
}
