//@ compile-flags: -Zmir-opt-level=0
//@ run-pass

#![feature(const_float_bits_conv)]
#![feature(const_float_classify)]
#![feature(f16)]
#![feature(f128)]
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

fn has_broken_floats() -> bool {
    // i586 targets are broken due to <https://github.com/rust-lang/rust/issues/114479>.
    std::env::var("TARGET").is_ok_and(|v| v.contains("i586"))
}

#[cfg(target_arch = "x86_64")]
fn f16(){
    const_assert!((1f16).to_bits(), 0x3c00);
    const_assert!(u16::from_be_bytes(1f16.to_be_bytes()), 0x3c00);
    const_assert!((12.5f16).to_bits(), 0x4a40);
    const_assert!(u16::from_le_bytes(12.5f16.to_le_bytes()), 0x4a40);
    const_assert!((1337f16).to_bits(), 0x6539);
    const_assert!(u16::from_ne_bytes(1337f16.to_ne_bytes()), 0x6539);
    const_assert!((-14.25f16).to_bits(), 0xcb20);
    const_assert!(f16::from_bits(0x3c00), 1.0);
    const_assert!(f16::from_be_bytes(0x3c00u16.to_be_bytes()), 1.0);
    const_assert!(f16::from_bits(0x4a40), 12.5);
    const_assert!(f16::from_le_bytes(0x4a40u16.to_le_bytes()), 12.5);
    const_assert!(f16::from_bits(0x5be0), 252.0);
    const_assert!(f16::from_ne_bytes(0x5be0u16.to_ne_bytes()), 252.0);
    const_assert!(f16::from_bits(0xcb20), -14.25);

    // Check that NaNs roundtrip their bits regardless of signalingness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    // NOTE: These names assume `f{BITS}::NAN` is a quiet NAN and IEEE754-2008's NaN rules apply!
    const QUIET_NAN: u16 = f16::NAN.to_bits() ^ 0x0155;
    const SIGNALING_NAN: u16 = f16::NAN.to_bits() ^ 0x02AA;

    const_assert!(f16::from_bits(QUIET_NAN).is_nan());
    const_assert!(f16::from_bits(SIGNALING_NAN).is_nan());
    const_assert!(f16::from_bits(QUIET_NAN).to_bits(), QUIET_NAN);
    if !has_broken_floats() {
        const_assert!(f16::from_bits(SIGNALING_NAN).to_bits(), SIGNALING_NAN);
    }
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
    // NOTE: These names assume `f{BITS}::NAN` is a quiet NAN and IEEE754-2008's NaN rules apply!
    const QUIET_NAN: u32 = f32::NAN.to_bits() ^ 0x002A_AAAA;
    const SIGNALING_NAN: u32 = f32::NAN.to_bits() ^ 0x0055_5555;

    const_assert!(f32::from_bits(QUIET_NAN).is_nan());
    const_assert!(f32::from_bits(SIGNALING_NAN).is_nan());
    const_assert!(f32::from_bits(QUIET_NAN).to_bits(), QUIET_NAN);
    if !has_broken_floats() {
        const_assert!(f32::from_bits(SIGNALING_NAN).to_bits(), SIGNALING_NAN);
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
    // NOTE: These names assume `f{BITS}::NAN` is a quiet NAN and IEEE754-2008's NaN rules apply!
    const QUIET_NAN: u64 = f64::NAN.to_bits() ^ 0x0005_5555_5555_5555;
    const SIGNALING_NAN: u64 = f64::NAN.to_bits() ^ 0x000A_AAAA_AAAA_AAAA;

    const_assert!(f64::from_bits(QUIET_NAN).is_nan());
    const_assert!(f64::from_bits(SIGNALING_NAN).is_nan());
    const_assert!(f64::from_bits(QUIET_NAN).to_bits(), QUIET_NAN);
    if !has_broken_floats() {
        const_assert!(f64::from_bits(SIGNALING_NAN).to_bits(), SIGNALING_NAN);
    }
}

#[cfg(target_arch = "x86_64")]
fn f128() {
    const_assert!((1f128).to_bits(), 0x3fff0000000000000000000000000000);
    const_assert!(u128::from_be_bytes(1f128.to_be_bytes()), 0x3fff0000000000000000000000000000);
    const_assert!((12.5f128).to_bits(), 0x40029000000000000000000000000000);
    const_assert!(u128::from_le_bytes(12.5f128.to_le_bytes()), 0x40029000000000000000000000000000);
    const_assert!((1337f128).to_bits(), 0x40094e40000000000000000000000000);
    const_assert!(u128::from_ne_bytes(1337f128.to_ne_bytes()), 0x40094e40000000000000000000000000);
    const_assert!((-14.25f128).to_bits(), 0xc002c800000000000000000000000000);
    const_assert!(f128::from_bits(0x3fff0000000000000000000000000000), 1.0);
    const_assert!(f128::from_be_bytes(0x3fff0000000000000000000000000000u128.to_be_bytes()), 1.0);
    const_assert!(f128::from_bits(0x40029000000000000000000000000000), 12.5);
    const_assert!(f128::from_le_bytes(0x40029000000000000000000000000000u128.to_le_bytes()), 12.5);
    const_assert!(f128::from_bits(0x40094e40000000000000000000000000), 1337.0);
    assert_eq!(f128::from_ne_bytes(0x40094e40000000000000000000000000u128.to_ne_bytes()), 1337.0);
    const_assert!(f128::from_bits(0xc002c800000000000000000000000000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signalingness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    // NOTE: These names assume `f{BITS}::NAN` is a quiet NAN and IEEE754-2008's NaN rules apply!
    const QUIET_NAN: u128 = f128::NAN.to_bits() | 0x0000_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA;
    const SIGNALING_NAN: u128 = f128::NAN.to_bits() ^ 0x0000_5555_5555_5555_5555_5555_5555_5555;

    const_assert!(f128::from_bits(QUIET_NAN).is_nan());
    const_assert!(f128::from_bits(SIGNALING_NAN).is_nan());
    const_assert!(f128::from_bits(QUIET_NAN).to_bits(), QUIET_NAN);
    if !has_broken_floats() {
        const_assert!(f128::from_bits(SIGNALING_NAN).to_bits(), SIGNALING_NAN);
    }
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        f16();
        f128();
    }
    f32();
    f64();
}
