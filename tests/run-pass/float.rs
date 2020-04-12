#![feature(track_caller)]
use std::fmt::Debug;

// Helper function to avoid promotion so that this tests "run-time" casts, not CTFE.
// Doesn't make a big difference when running this in Miri, but it means we can compare this
// with the LLVM backend by running `rustc -Zmir-opt-level=0 -Zsaturating-float-casts`.
#[track_caller]
#[inline(never)]
fn assert_eq<T: PartialEq + Debug>(x: T, y: T) {
    assert_eq!(x, y);
}

trait FloatToInt<Int>: Copy {
    fn cast(self) -> Int;
    unsafe fn cast_unchecked(self) -> Int;
}

impl FloatToInt<i8> for f32 {
    fn cast(self) -> i8 { self as _ }
    unsafe fn cast_unchecked(self) -> i8 { self.to_int_unchecked() }
}
impl FloatToInt<i32> for f32 {
    fn cast(self) -> i32 { self as _ }
    unsafe fn cast_unchecked(self) -> i32 { self.to_int_unchecked() }
}
impl FloatToInt<u32> for f32 {
    fn cast(self) -> u32 { self as _ }
    unsafe fn cast_unchecked(self) -> u32 { self.to_int_unchecked() }
}
impl FloatToInt<i64> for f32 {
    fn cast(self) -> i64 { self as _ }
    unsafe fn cast_unchecked(self) -> i64 { self.to_int_unchecked() }
}
impl FloatToInt<u64> for f32 {
    fn cast(self) -> u64 { self as _ }
    unsafe fn cast_unchecked(self) -> u64 { self.to_int_unchecked() }
}

impl FloatToInt<i8> for f64 {
    fn cast(self) -> i8 { self as _ }
    unsafe fn cast_unchecked(self) -> i8 { self.to_int_unchecked() }
}
impl FloatToInt<i32> for f64 {
    fn cast(self) -> i32 { self as _ }
    unsafe fn cast_unchecked(self) -> i32 { self.to_int_unchecked() }
}
impl FloatToInt<u32> for f64 {
    fn cast(self) -> u32 { self as _ }
    unsafe fn cast_unchecked(self) -> u32 { self.to_int_unchecked() }
}
impl FloatToInt<i64> for f64 {
    fn cast(self) -> i64 { self as _ }
    unsafe fn cast_unchecked(self) -> i64 { self.to_int_unchecked() }
}
impl FloatToInt<u64> for f64 {
    fn cast(self) -> u64 { self as _ }
    unsafe fn cast_unchecked(self) -> u64 { self.to_int_unchecked() }
}

/// Test this cast both via `as` and via `approx_unchecked` (i.e., it must not saturate).
#[track_caller]
#[inline(never)]
fn test_cast<F, I>(x: F, y: I)
    where F: FloatToInt<I>, I: PartialEq + Debug
{
    assert_eq!(x.cast(), y);
    assert_eq!(unsafe { x.cast_unchecked() }, y);
}

fn main() {
    basic();
    casts();
    ops();
}

fn basic() {
    // basic arithmetic
    assert_eq(6.0_f32*6.0_f32, 36.0_f32);
    assert_eq(6.0_f64*6.0_f64, 36.0_f64);
    assert_eq(-{5.0_f32}, -5.0_f32);
    assert_eq(-{5.0_f64}, -5.0_f64);
    // infinities, NaN
    assert!((5.0_f32/0.0).is_infinite());
    assert_ne!({5.0_f32/0.0}, {-5.0_f32/0.0});
    assert!((5.0_f64/0.0).is_infinite());
    assert_ne!({5.0_f64/0.0}, {5.0_f64/-0.0});
    assert!((-5.0_f32).sqrt().is_nan());
    assert!((-5.0_f64).sqrt().is_nan());
    assert_ne!(f32::NAN, f32::NAN);
    assert_ne!(f64::NAN, f64::NAN);
    // negative zero
    let posz = 0.0f32;
    let negz = -0.0f32;
    assert_eq(posz, negz);
    assert_ne!(posz.to_bits(), negz.to_bits());
    let posz = 0.0f64;
    let negz = -0.0f64;
    assert_eq(posz, negz);
    assert_ne!(posz.to_bits(), negz.to_bits());
    // byte-level transmute
    let x: u64 = unsafe { std::mem::transmute(42.0_f64) };
    let y: f64 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f64);
    let x: u32 = unsafe { std::mem::transmute(42.0_f32) };
    let y: f32 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f32);
}

fn casts() {
    // f32 -> i8
    test_cast::<f32, i8>(127.99, 127);
    test_cast::<f32, i8>(-128.99, -128);

    // f32 -> i32
    test_cast::<f32, i32>(0.0, 0);
    test_cast::<f32, i32>(-0.0, 0);
    test_cast::<f32, i32>(/*0x1p-149*/ f32::from_bits(0x00000001), 0);
    test_cast::<f32, i32>(/*-0x1p-149*/ f32::from_bits(0x80000001), 0);
    test_cast::<f32, i32>(/*0x1.19999ap+0*/ f32::from_bits(0x3f8ccccd), 1);
    test_cast::<f32, i32>(/*-0x1.19999ap+0*/ f32::from_bits(0xbf8ccccd), -1);
    test_cast::<f32, i32>(1.9, 1);
    test_cast::<f32, i32>(-1.9, -1);
    test_cast::<f32, i32>(5.0, 5);
    test_cast::<f32, i32>(-5.0, -5);
    test_cast::<f32, i32>(2147483520.0, 2147483520);
    test_cast::<f32, i32>(-2147483648.0, -2147483648);
    // unrepresentable casts
    assert_eq::<i32>(2147483648.0f32 as i32, i32::MAX);
    assert_eq::<i32>(-2147483904.0f32 as i32, i32::MIN);
    assert_eq::<i32>(f32::MAX as i32, i32::MAX);
    assert_eq::<i32>(f32::MIN as i32, i32::MIN);
    assert_eq::<i32>(f32::INFINITY as i32, i32::MAX);
    assert_eq::<i32>(f32::NEG_INFINITY as i32, i32::MIN);
    assert_eq::<i32>(f32::NAN as i32, 0);
    assert_eq::<i32>((-f32::NAN) as i32, 0);

    // f32 -> u32
    test_cast::<f32, u32>(0.0, 0);
    test_cast::<f32, u32>(-0.0, 0);
    test_cast::<f32, u32>(/*0x1p-149*/ f32::from_bits(0x1), 0);
    test_cast::<f32, u32>(/*-0x1p-149*/ f32::from_bits(0x80000001), 0);
    test_cast::<f32, u32>(/*0x1.19999ap+0*/ f32::from_bits(0x3f8ccccd), 1);
    test_cast::<f32, u32>(1.9, 1);
    test_cast::<f32, u32>(5.0, 5);
    test_cast::<f32, u32>(2147483648.0, 0x8000_0000);
    test_cast::<f32, u32>(4294967040.0, 0u32.wrapping_sub(256));
    test_cast::<f32, u32>(/*-0x1.ccccccp-1*/ f32::from_bits(0xbf666666), 0);
    test_cast::<f32, u32>(/*-0x1.fffffep-1*/ f32::from_bits(0xbf7fffff), 0);
    test_cast::<f32, u32>((u32::MAX-127) as f32, u32::MAX); // rounding loss
    test_cast::<f32, u32>((u32::MAX-128) as f32, u32::MAX-255); // rounding loss
    // unrepresentable casts
    assert_eq::<u32>(4294967296.0f32 as u32, u32::MAX);
    assert_eq::<u32>(-5.0f32 as u32, 0);
    assert_eq::<u32>(f32::MAX as u32, u32::MAX);
    assert_eq::<u32>(f32::MIN as u32, 0);
    assert_eq::<u32>(f32::INFINITY as u32, u32::MAX);
    assert_eq::<u32>(f32::NEG_INFINITY as u32, 0);
    assert_eq::<u32>(f32::NAN as u32, 0);
    assert_eq::<u32>((-f32::NAN) as u32, 0);

    // f32 -> i64
    test_cast::<f32, i64>(4294967296.0, 4294967296);
    test_cast::<f32, i64>(-4294967296.0, -4294967296);
    test_cast::<f32, i64>(9223371487098961920.0, 9223371487098961920);
    test_cast::<f32, i64>(-9223372036854775808.0, -9223372036854775808);

    // f64 -> i8
    test_cast::<f64, i8>(127.99, 127);
    test_cast::<f64, i8>(-128.99, -128);

    // f64 -> i32
    test_cast::<f64, i32>(0.0, 0);
    test_cast::<f64, i32>(-0.0, 0);
    test_cast::<f64, i32>(/*0x1.199999999999ap+0*/ f64::from_bits(0x3ff199999999999a), 1);
    test_cast::<f64, i32>(/*-0x1.199999999999ap+0*/ f64::from_bits(0xbff199999999999a), -1);
    test_cast::<f64, i32>(1.9, 1);
    test_cast::<f64, i32>(-1.9, -1);
    test_cast::<f64, i32>(1e8, 100_000_000);
    test_cast::<f64, i32>(2147483647.0, 2147483647);
    test_cast::<f64, i32>(-2147483648.0, -2147483648);
    // unrepresentable casts
    assert_eq::<i32>(2147483648.0f64 as i32, i32::MAX);
    assert_eq::<i32>(-2147483649.0f64 as i32, i32::MIN);

    // f64 -> i64
    test_cast::<f64, i64>(0.0, 0);
    test_cast::<f64, i64>(-0.0, 0);
    test_cast::<f64, i64>(/*0x0.0000000000001p-1022*/ f64::from_bits(0x1), 0);
    test_cast::<f64, i64>(/*-0x0.0000000000001p-1022*/ f64::from_bits(0x8000000000000001), 0);
    test_cast::<f64, i64>(/*0x1.199999999999ap+0*/ f64::from_bits(0x3ff199999999999a), 1);
    test_cast::<f64, i64>(/*-0x1.199999999999ap+0*/ f64::from_bits(0xbff199999999999a), -1);
    test_cast::<f64, i64>(5.0, 5);
    test_cast::<f64, i64>(5.9, 5);
    test_cast::<f64, i64>(-5.0, -5);
    test_cast::<f64, i64>(-5.9, -5);
    test_cast::<f64, i64>(4294967296.0, 4294967296);
    test_cast::<f64, i64>(-4294967296.0, -4294967296);
    test_cast::<f64, i64>(9223372036854774784.0, 9223372036854774784);
    test_cast::<f64, i64>(-9223372036854775808.0, -9223372036854775808);
    // unrepresentable casts
    assert_eq::<i64>(9223372036854775808.0f64 as i64, i64::MAX);
    assert_eq::<i64>(-9223372036854777856.0f64 as i64, i64::MIN);
    assert_eq::<i64>(f64::MAX as i64, i64::MAX);
    assert_eq::<i64>(f64::MIN as i64, i64::MIN);
    assert_eq::<i64>(f64::INFINITY as i64, i64::MAX);
    assert_eq::<i64>(f64::NEG_INFINITY as i64, i64::MIN);
    assert_eq::<i64>(f64::NAN as i64, 0);
    assert_eq::<i64>((-f64::NAN) as i64, 0);

    // f64 -> u64
    test_cast::<f64, u64>(0.0, 0);
    test_cast::<f64, u64>(-0.0, 0);
    test_cast::<f64, u64>(5.0, 5);
    test_cast::<f64, u64>(-5.0, 0);
    test_cast::<f64, u64>(1e16, 10000000000000000);
    test_cast::<f64, u64>((u64::MAX-1023) as f64, u64::MAX); // rounding loss
    test_cast::<f64, u64>((u64::MAX-1024) as f64, u64::MAX-2047); // rounding loss
    test_cast::<f64, u64>(9223372036854775808.0, 9223372036854775808);
    // unrepresentable casts
    assert_eq::<u64>(18446744073709551616.0f64 as u64, u64::MAX);
    assert_eq::<u64>(f64::MAX as u64, u64::MAX);
    assert_eq::<u64>(f64::MIN as u64, 0);
    assert_eq::<u64>(f64::INFINITY as u64, u64::MAX);
    assert_eq::<u64>(f64::NEG_INFINITY as u64, 0);
    assert_eq::<u64>(f64::NAN as u64, 0);
    assert_eq::<u64>((-f64::NAN) as u64, 0);

    // int -> f32
    assert_eq::<f32>(127i8 as f32, 127.0);
    assert_eq::<f32>(2147483647i32 as f32, 2147483648.0);
    assert_eq::<f32>((-2147483648i32) as f32, -2147483648.0);
    assert_eq::<f32>(1234567890i32 as f32, /*0x1.26580cp+30*/ f32::from_bits(0x4e932c06));
    assert_eq::<f32>(16777217i32 as f32, 16777216.0);
    assert_eq::<f32>((-16777217i32) as f32, -16777216.0);
    assert_eq::<f32>(16777219i32 as f32, 16777220.0);
    assert_eq::<f32>((-16777219i32) as f32, -16777220.0);
    assert_eq::<f32>(0x7fffff4000000001i64 as f32, /*0x1.fffffep+62*/ f32::from_bits(0x5effffff));
    assert_eq::<f32>(0x8000004000000001u64 as i64 as f32, /*-0x1.fffffep+62*/ f32::from_bits(0xdeffffff));
    assert_eq::<f32>(0x0020000020000001i64 as f32, /*0x1.000002p+53*/ f32::from_bits(0x5a000001));
    assert_eq::<f32>(0xffdfffffdfffffffu64 as i64 as f32, /*-0x1.000002p+53*/ f32::from_bits(0xda000001));
    assert_eq::<f32>(i128::MIN as f32, -170141183460469231731687303715884105728.0f32);
    assert_eq::<f32>(u128::MAX as f32, f32::INFINITY); // saturation

    // int -> f64
    assert_eq::<f64>(127i8 as f64, 127.0);
    assert_eq::<f64>(i16::MIN as f64, -32768.0f64);
    assert_eq::<f64>(2147483647i32 as f64, 2147483647.0);
    assert_eq::<f64>(-2147483648i32 as f64, -2147483648.0);
    assert_eq::<f64>(987654321i32 as f64, 987654321.0);
    assert_eq::<f64>(9223372036854775807i64 as f64, 9223372036854775807.0);
    assert_eq::<f64>(-9223372036854775808i64 as f64, -9223372036854775808.0);
    assert_eq::<f64>(4669201609102990i64 as f64, 4669201609102990.0); // Feigenbaum (?)
    assert_eq::<f64>(9007199254740993i64 as f64, 9007199254740992.0);
    assert_eq::<f64>(-9007199254740993i64 as f64, -9007199254740992.0);
    assert_eq::<f64>(9007199254740995i64 as f64, 9007199254740996.0);
    assert_eq::<f64>(-9007199254740995i64 as f64, -9007199254740996.0);
    assert_eq::<f64>(u128::MAX as f64, 340282366920938463463374607431768211455.0f64); // even that fits...

    // f32 -> f64
    assert_eq::<u64>((0.0f32 as f64).to_bits(), 0.0f64.to_bits());
    assert_eq::<u64>(((-0.0f32) as f64).to_bits(), (-0.0f64).to_bits());
    assert_eq::<f64>(5.0f32 as f64, 5.0f64);
    assert_eq::<f64>(/*0x1p-149*/ f32::from_bits(0x1) as f64, /*0x1p-149*/ f64::from_bits(0x36a0000000000000));
    assert_eq::<f64>(/*-0x1p-149*/ f32::from_bits(0x80000001) as f64, /*-0x1p-149*/ f64::from_bits(0xb6a0000000000000));
    assert_eq::<f64>(/*0x1.fffffep+127*/ f32::from_bits(0x7f7fffff) as f64, /*0x1.fffffep+127*/ f64::from_bits(0x47efffffe0000000));
    assert_eq::<f64>(/*-0x1.fffffep+127*/ (-f32::from_bits(0x7f7fffff)) as f64, /*-0x1.fffffep+127*/ -f64::from_bits(0x47efffffe0000000));
    assert_eq::<f64>(/*0x1p-119*/ f32::from_bits(0x4000000) as f64, /*0x1p-119*/ f64::from_bits(0x3880000000000000));
    assert_eq::<f64>(/*0x1.8f867ep+125*/ f32::from_bits(0x7e47c33f) as f64, 6.6382536710104395e+37);
    assert_eq::<f64>(f32::INFINITY as f64, f64::INFINITY);
    assert_eq::<f64>(f32::NEG_INFINITY as f64, f64::NEG_INFINITY);

    // f64 -> f32
    assert_eq::<u32>((0.0f64 as f32).to_bits(), 0.0f32.to_bits());
    assert_eq::<u32>(((-0.0f64) as f32).to_bits(), (-0.0f32).to_bits());
    assert_eq::<f32>(5.0f64 as f32, 5.0f32);
    assert_eq::<f32>(/*0x0.0000000000001p-1022*/ f64::from_bits(0x1) as f32, 0.0);
    assert_eq::<f32>(/*-0x0.0000000000001p-1022*/ (-f64::from_bits(0x1)) as f32, -0.0);

    assert_eq::<f32>(/*0x1.fffffe0000000p-127*/ f64::from_bits(0x380fffffe0000000) as f32, /*0x1p-149*/ f32::from_bits(0x800000));
    assert_eq::<f32>(/*0x1.4eae4f7024c7p+108*/ f64::from_bits(0x46b4eae4f7024c70) as f32, /*0x1.4eae5p+108*/ f32::from_bits(0x75a75728));

    assert_eq::<f32>(f64::MAX as f32, f32::INFINITY);
    assert_eq::<f32>(f64::MIN as f32, f32::NEG_INFINITY);
    assert_eq::<f32>(f64::INFINITY as f32, f32::INFINITY);
    assert_eq::<f32>(f64::NEG_INFINITY as f32, f32::NEG_INFINITY);
}

fn ops() {
    // f32 min/max
    assert_eq((1.0 as f32).max(-1.0), 1.0);
    assert_eq((1.0 as f32).min(-1.0), -1.0);
    assert_eq(f32::NAN.min(9.0), 9.0);
    assert_eq(f32::NAN.max(-9.0), -9.0);
    assert_eq((9.0 as f32).min(f32::NAN), 9.0);
    assert_eq((-9.0 as f32).max(f32::NAN), -9.0);

    // f64 min/max
    assert_eq((1.0 as f64).max(-1.0), 1.0);
    assert_eq((1.0 as f64).min(-1.0), -1.0);
    assert_eq(f64::NAN.min(9.0), 9.0);
    assert_eq(f64::NAN.max(-9.0), -9.0);
    assert_eq((9.0 as f64).min(f64::NAN), 9.0);
    assert_eq((-9.0 as f64).max(f64::NAN), -9.0);

    // f32 copysign
    assert_eq(3.5_f32.copysign(0.42), 3.5_f32);
    assert_eq(3.5_f32.copysign(-0.42), -3.5_f32);
    assert_eq((-3.5_f32).copysign(0.42), 3.5_f32);
    assert_eq((-3.5_f32).copysign(-0.42), -3.5_f32);
    assert!(f32::NAN.copysign(1.0).is_nan());

    // f64 copysign
    assert_eq(3.5_f64.copysign(0.42), 3.5_f64);
    assert_eq(3.5_f64.copysign(-0.42), -3.5_f64);
    assert_eq((-3.5_f64).copysign(0.42), 3.5_f64);
    assert_eq((-3.5_f64).copysign(-0.42), -3.5_f64);
    assert!(f64::NAN.copysign(1.0).is_nan());
}
