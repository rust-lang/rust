//@ ignore-auxiliary (used by `./saturating-float-casts.rs` and `./saturating-float-casts-wasm.rs`)

// Tests saturating float->int casts. See u128-as-f32.rs for the opposite direction.
//
// Some of these tests come from a similar file in miri,
// tests/run-pass/float.rs. Individual test cases are potentially duplicated
// with the previously existing tests, but since this runs so quickly anyway,
// we're not spending the time to figure out exactly which ones should be
// merged.

extern crate test;

use self::test::black_box;

macro_rules! test {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => (
        // black_box disables constant evaluation to test run-time conversions:
        assert_eq!(black_box::<$src_ty>($val) as $dest_ty, $expected,
                    "run-time {} -> {}", stringify!($src_ty), stringify!($dest_ty));

        {
            const X: $src_ty = $val;
            const Y: $dest_ty = X as $dest_ty;
            assert_eq!(Y, $expected,
                        "const eval {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        }
    );

    ($fval:expr, f* -> $ity:ident, $ival:expr) => (
        test!($fval, f32 -> $ity, $ival);
        test!($fval, f64 -> $ity, $ival);
    )
}

macro_rules! common_fptoi_tests {
    ($fty:ident -> $($ity:ident)+) => ({ $(
        test!($fty::NAN, $fty -> $ity, 0);
        test!($fty::INFINITY, $fty -> $ity, $ity::MAX);
        test!($fty::NEG_INFINITY, $fty -> $ity, $ity::MIN);
        // These two tests are not solely float->int tests, in particular the latter relies on
        // `u128::MAX as f32` not being UB. But that's okay, since this file tests int->float
        // as well, the test is just slightly misplaced.
        test!($ity::MIN as $fty, $fty -> $ity, $ity::MIN);
        test!($ity::MAX as $fty, $fty -> $ity, $ity::MAX);
        test!(0., $fty -> $ity, 0);
        test!($fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.9, $fty -> $ity, 0);
        test!(1., $fty -> $ity, 1);
        test!(42., $fty -> $ity, 42);
    )+ });

    (f* -> $($ity:ident)+) => ({
        common_fptoi_tests!(f32 -> $($ity)+);
        common_fptoi_tests!(f64 -> $($ity)+);
    })
}

macro_rules! fptoui_tests {
    ($fty: ident -> $($ity: ident)+) => ({ $(
        test!(-0., $fty -> $ity, 0);
        test!(-$fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.99999994, $fty -> $ity, 0);
        test!(-1., $fty -> $ity, 0);
        test!(-100., $fty -> $ity, 0);
        test!(#[allow(overflowing_literals)] -1e50, $fty -> $ity, 0);
        test!(#[allow(overflowing_literals)] -1e130, $fty -> $ity, 0);
    )+ });

    (f* -> $($ity:ident)+) => ({
        fptoui_tests!(f32 -> $($ity)+);
        fptoui_tests!(f64 -> $($ity)+);
    })
}

use std::fmt::Debug;

// Helper function to avoid promotion so that this tests "run-time" casts, not CTFE.
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
    fn cast(self) -> i8 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i8 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<i32> for f32 {
    fn cast(self) -> i32 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i32 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<u32> for f32 {
    fn cast(self) -> u32 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> u32 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<i64> for f32 {
    fn cast(self) -> i64 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i64 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<u64> for f32 {
    fn cast(self) -> u64 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> u64 {
        self.to_int_unchecked()
    }
}

impl FloatToInt<i8> for f64 {
    fn cast(self) -> i8 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i8 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<i32> for f64 {
    fn cast(self) -> i32 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i32 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<u32> for f64 {
    fn cast(self) -> u32 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> u32 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<i64> for f64 {
    fn cast(self) -> i64 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i64 {
        self.to_int_unchecked()
    }
}
impl FloatToInt<u64> for f64 {
    fn cast(self) -> u64 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> u64 {
        self.to_int_unchecked()
    }
}
// FIXME emscripten does not support i128
#[cfg(not(target_os = "emscripten"))]
impl FloatToInt<i128> for f64 {
    fn cast(self) -> i128 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> i128 {
        self.to_int_unchecked()
    }
}
// FIXME emscripten does not support i128
#[cfg(not(target_os = "emscripten"))]
impl FloatToInt<u128> for f64 {
    fn cast(self) -> u128 {
        self as _
    }
    unsafe fn cast_unchecked(self) -> u128 {
        self.to_int_unchecked()
    }
}

/// Test this cast both via `as` and via `to_int_unchecked` (i.e., it must not saturate).
#[track_caller]
#[inline(never)]
fn test_both_cast<F, I>(x: F, y: I)
where
    F: FloatToInt<I>,
    I: PartialEq + Debug,
{
    assert_eq!(x.cast(), y);
    assert_eq!(unsafe { x.cast_unchecked() }, y);
}

fn casts() {
    // f32 -> i8
    test_both_cast::<f32, i8>(127.99, 127);
    test_both_cast::<f32, i8>(-128.99, -128);

    // f32 -> i32
    test_both_cast::<f32, i32>(0.0, 0);
    test_both_cast::<f32, i32>(-0.0, 0);
    test_both_cast::<f32, i32>(/*0x1p-149*/ f32::from_bits(0x00000001), 0);
    test_both_cast::<f32, i32>(/*-0x1p-149*/ f32::from_bits(0x80000001), 0);
    test_both_cast::<f32, i32>(/*0x1.19999ap+0*/ f32::from_bits(0x3f8ccccd), 1);
    test_both_cast::<f32, i32>(/*-0x1.19999ap+0*/ f32::from_bits(0xbf8ccccd), -1);
    test_both_cast::<f32, i32>(1.9, 1);
    test_both_cast::<f32, i32>(-1.9, -1);
    test_both_cast::<f32, i32>(5.0, 5);
    test_both_cast::<f32, i32>(-5.0, -5);
    test_both_cast::<f32, i32>(2147483520.0, 2147483520);
    test_both_cast::<f32, i32>(-2147483648.0, -2147483648);
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
    test_both_cast::<f32, u32>(0.0, 0);
    test_both_cast::<f32, u32>(-0.0, 0);
    test_both_cast::<f32, u32>(-0.9999999, 0);
    test_both_cast::<f32, u32>(/*0x1p-149*/ f32::from_bits(0x1), 0);
    test_both_cast::<f32, u32>(/*-0x1p-149*/ f32::from_bits(0x80000001), 0);
    test_both_cast::<f32, u32>(/*0x1.19999ap+0*/ f32::from_bits(0x3f8ccccd), 1);
    test_both_cast::<f32, u32>(1.9, 1);
    test_both_cast::<f32, u32>(5.0, 5);
    test_both_cast::<f32, u32>(2147483648.0, 0x8000_0000);
    test_both_cast::<f32, u32>(4294967040.0, 0u32.wrapping_sub(256));
    test_both_cast::<f32, u32>(/*-0x1.ccccccp-1*/ f32::from_bits(0xbf666666), 0);
    test_both_cast::<f32, u32>(/*-0x1.fffffep-1*/ f32::from_bits(0xbf7fffff), 0);
    test_both_cast::<f32, u32>((u32::MAX - 128) as f32, u32::MAX - 255); // rounding loss

    // unrepresentable casts:

    // rounds up and then becomes unrepresentable
    assert_eq::<u32>((u32::MAX - 127) as f32 as u32, u32::MAX);

    assert_eq::<u32>(4294967296.0f32 as u32, u32::MAX);
    assert_eq::<u32>(-5.0f32 as u32, 0);
    assert_eq::<u32>(f32::MAX as u32, u32::MAX);
    assert_eq::<u32>(f32::MIN as u32, 0);
    assert_eq::<u32>(f32::INFINITY as u32, u32::MAX);
    assert_eq::<u32>(f32::NEG_INFINITY as u32, 0);
    assert_eq::<u32>(f32::NAN as u32, 0);
    assert_eq::<u32>((-f32::NAN) as u32, 0);

    // f32 -> i64
    test_both_cast::<f32, i64>(4294967296.0, 4294967296);
    test_both_cast::<f32, i64>(-4294967296.0, -4294967296);
    test_both_cast::<f32, i64>(9223371487098961920.0, 9223371487098961920);
    test_both_cast::<f32, i64>(-9223372036854775808.0, -9223372036854775808);

    // f64 -> i8
    test_both_cast::<f64, i8>(127.99, 127);
    test_both_cast::<f64, i8>(-128.99, -128);

    // f64 -> i32
    test_both_cast::<f64, i32>(0.0, 0);
    test_both_cast::<f64, i32>(-0.0, 0);
    test_both_cast::<f64, i32>(/*0x1.199999999999ap+0*/ f64::from_bits(0x3ff199999999999a), 1);
    test_both_cast::<f64, i32>(
        /*-0x1.199999999999ap+0*/ f64::from_bits(0xbff199999999999a),
        -1,
    );
    test_both_cast::<f64, i32>(1.9, 1);
    test_both_cast::<f64, i32>(-1.9, -1);
    test_both_cast::<f64, i32>(1e8, 100_000_000);
    test_both_cast::<f64, i32>(2147483647.0, 2147483647);
    test_both_cast::<f64, i32>(-2147483648.0, -2147483648);
    // unrepresentable casts
    assert_eq::<i32>(2147483648.0f64 as i32, i32::MAX);
    assert_eq::<i32>(-2147483649.0f64 as i32, i32::MIN);

    // f64 -> i64
    test_both_cast::<f64, i64>(0.0, 0);
    test_both_cast::<f64, i64>(-0.0, 0);
    test_both_cast::<f64, i64>(/*0x0.0000000000001p-1022*/ f64::from_bits(0x1), 0);
    test_both_cast::<f64, i64>(
        /*-0x0.0000000000001p-1022*/ f64::from_bits(0x8000000000000001),
        0,
    );
    test_both_cast::<f64, i64>(/*0x1.199999999999ap+0*/ f64::from_bits(0x3ff199999999999a), 1);
    test_both_cast::<f64, i64>(
        /*-0x1.199999999999ap+0*/ f64::from_bits(0xbff199999999999a),
        -1,
    );
    test_both_cast::<f64, i64>(5.0, 5);
    test_both_cast::<f64, i64>(5.9, 5);
    test_both_cast::<f64, i64>(-5.0, -5);
    test_both_cast::<f64, i64>(-5.9, -5);
    test_both_cast::<f64, i64>(4294967296.0, 4294967296);
    test_both_cast::<f64, i64>(-4294967296.0, -4294967296);
    test_both_cast::<f64, i64>(9223372036854774784.0, 9223372036854774784);
    test_both_cast::<f64, i64>(-9223372036854775808.0, -9223372036854775808);
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
    test_both_cast::<f64, u64>(0.0, 0);
    test_both_cast::<f64, u64>(-0.0, 0);
    test_both_cast::<f64, u64>(-0.99999999999, 0);
    test_both_cast::<f64, u64>(5.0, 5);
    test_both_cast::<f64, u64>(1e16, 10000000000000000);
    test_both_cast::<f64, u64>((u64::MAX - 1024) as f64, u64::MAX - 2047); // rounding loss
    test_both_cast::<f64, u64>(9223372036854775808.0, 9223372036854775808);
    // unrepresentable casts
    assert_eq::<u64>(-5.0f64 as u64, 0);
    // rounds up and then becomes unrepresentable
    assert_eq::<u64>((u64::MAX - 1023) as f64 as u64, u64::MAX);
    assert_eq::<u64>(18446744073709551616.0f64 as u64, u64::MAX);
    assert_eq::<u64>(f64::MAX as u64, u64::MAX);
    assert_eq::<u64>(f64::MIN as u64, 0);
    assert_eq::<u64>(f64::INFINITY as u64, u64::MAX);
    assert_eq::<u64>(f64::NEG_INFINITY as u64, 0);
    assert_eq::<u64>(f64::NAN as u64, 0);
    assert_eq::<u64>((-f64::NAN) as u64, 0);

    // FIXME emscripten does not support i128
    #[cfg(not(target_os = "emscripten"))]
    {
        // f64 -> i128
        assert_eq::<i128>(f64::MAX as i128, i128::MAX);
        assert_eq::<i128>(f64::MIN as i128, i128::MIN);

        // f64 -> u128
        assert_eq::<u128>(f64::MAX as u128, u128::MAX);
        assert_eq::<u128>(f64::MIN as u128, 0);
    }

    // int -> f32
    assert_eq::<f32>(127i8 as f32, 127.0);
    assert_eq::<f32>(2147483647i32 as f32, 2147483648.0);
    assert_eq::<f32>((-2147483648i32) as f32, -2147483648.0);
    assert_eq::<f32>(1234567890i32 as f32, /*0x1.26580cp+30*/ f32::from_bits(0x4e932c06));
    assert_eq::<f32>(16777217i32 as f32, 16777216.0);
    assert_eq::<f32>((-16777217i32) as f32, -16777216.0);
    assert_eq::<f32>(16777219i32 as f32, 16777220.0);
    assert_eq::<f32>((-16777219i32) as f32, -16777220.0);
    assert_eq::<f32>(
        0x7fffff4000000001i64 as f32,
        /*0x1.fffffep+62*/ f32::from_bits(0x5effffff),
    );
    assert_eq::<f32>(
        0x8000004000000001u64 as i64 as f32,
        /*-0x1.fffffep+62*/ f32::from_bits(0xdeffffff),
    );
    assert_eq::<f32>(
        0x0020000020000001i64 as f32,
        /*0x1.000002p+53*/ f32::from_bits(0x5a000001),
    );
    assert_eq::<f32>(
        0xffdfffffdfffffffu64 as i64 as f32,
        /*-0x1.000002p+53*/ f32::from_bits(0xda000001),
    );
    // FIXME emscripten does not support i128
    #[cfg(not(target_os = "emscripten"))]
    {
        assert_eq::<f32>(i128::MIN as f32, -170141183460469231731687303715884105728.0f32);
        assert_eq::<f32>(u128::MAX as f32, f32::INFINITY); // saturation
    }

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
    // FIXME emscripten does not support i128
    #[cfg(not(target_os = "emscripten"))]
    {
        // even that fits...
        assert_eq::<f64>(u128::MAX as f64, 340282366920938463463374607431768211455.0f64);
    }

    // f32 -> f64
    assert_eq::<u64>((0.0f32 as f64).to_bits(), 0.0f64.to_bits());
    assert_eq::<u64>(((-0.0f32) as f64).to_bits(), (-0.0f64).to_bits());
    assert_eq::<f64>(5.0f32 as f64, 5.0f64);
    assert_eq::<f64>(
        /*0x1p-149*/ f32::from_bits(0x1) as f64,
        /*0x1p-149*/ f64::from_bits(0x36a0000000000000),
    );
    assert_eq::<f64>(
        /*-0x1p-149*/ f32::from_bits(0x80000001) as f64,
        /*-0x1p-149*/ f64::from_bits(0xb6a0000000000000),
    );
    assert_eq::<f64>(
        /*0x1.fffffep+127*/ f32::from_bits(0x7f7fffff) as f64,
        /*0x1.fffffep+127*/ f64::from_bits(0x47efffffe0000000),
    );
    assert_eq::<f64>(
        /*-0x1.fffffep+127*/ (-f32::from_bits(0x7f7fffff)) as f64,
        /*-0x1.fffffep+127*/ -f64::from_bits(0x47efffffe0000000),
    );
    assert_eq::<f64>(
        /*0x1p-119*/ f32::from_bits(0x4000000) as f64,
        /*0x1p-119*/ f64::from_bits(0x3880000000000000),
    );
    assert_eq::<f64>(
        /*0x1.8f867ep+125*/ f32::from_bits(0x7e47c33f) as f64,
        6.6382536710104395e+37,
    );
    assert_eq::<f64>(f32::INFINITY as f64, f64::INFINITY);
    assert_eq::<f64>(f32::NEG_INFINITY as f64, f64::NEG_INFINITY);

    // f64 -> f32
    assert_eq::<u32>((0.0f64 as f32).to_bits(), 0.0f32.to_bits());
    assert_eq::<u32>(((-0.0f64) as f32).to_bits(), (-0.0f32).to_bits());
    assert_eq::<f32>(5.0f64 as f32, 5.0f32);
    assert_eq::<f32>(/*0x0.0000000000001p-1022*/ f64::from_bits(0x1) as f32, 0.0);
    assert_eq::<f32>(/*-0x0.0000000000001p-1022*/ (-f64::from_bits(0x1)) as f32, -0.0);
    assert_eq::<f32>(
        /*0x1.fffffe0000000p-127*/ f64::from_bits(0x380fffffe0000000) as f32,
        /*0x1p-149*/ f32::from_bits(0x800000),
    );
    assert_eq::<f32>(
        /*0x1.4eae4f7024c7p+108*/ f64::from_bits(0x46b4eae4f7024c70) as f32,
        /*0x1.4eae5p+108*/ f32::from_bits(0x75a75728),
    );
    assert_eq::<f32>(f64::MAX as f32, f32::INFINITY);
    assert_eq::<f32>(f64::MIN as f32, f32::NEG_INFINITY);
    assert_eq::<f32>(f64::INFINITY as f32, f32::INFINITY);
    assert_eq::<f32>(f64::NEG_INFINITY as f32, f32::NEG_INFINITY);
}

pub fn run() {
    casts(); // from miri's tests

    common_fptoi_tests!(f* -> i8 i16 i32 i64 u8 u16 u32 u64);
    fptoui_tests!(f* -> u8 u16 u32 u64);
    // FIXME emscripten does not support i128
    #[cfg(not(target_os = "emscripten"))]
    {
        common_fptoi_tests!(f* -> i128 u128);
        fptoui_tests!(f* -> u128);
    }

    // The following tests cover edge cases for some integer types.

    // # u8
    test!(254., f* -> u8, 254);
    test!(256., f* -> u8, 255);

    // # i8
    test!(-127., f* -> i8, -127);
    test!(-129., f* -> i8, -128);
    test!(126., f* -> i8, 126);
    test!(128., f* -> i8, 127);

    // # i32
    // -2147483648. is i32::MIN (exactly)
    test!(-2147483648., f* -> i32, i32::MIN);
    // 2147483648. is i32::MAX rounded up
    test!(2147483648., f32 -> i32, 2147483647);
    // With 24 significand bits, floats with magnitude in [2^30 + 1, 2^31] are rounded to
    // multiples of 2^7. Therefore, nextDown(round(i32::MAX)) is 2^31 - 128:
    test!(2147483520., f32 -> i32, 2147483520);
    // Similarly, nextUp(i32::MIN) is i32::MIN + 2^8 and nextDown(i32::MIN) is i32::MIN - 2^7
    test!(-2147483904., f* -> i32, i32::MIN);
    test!(-2147483520., f* -> i32, -2147483520);

    // # u32
    // round(MAX) and nextUp(round(MAX))
    test!(4294967040., f* -> u32, 4294967040);
    test!(4294967296., f* -> u32, 4294967295);

    // # u128
    #[cfg(not(target_os = "emscripten"))]
    {
        // float->int:
        test!(f32::MAX, f32 -> u128, 0xffffff00000000000000000000000000);
        // nextDown(f32::MAX) = 2^128 - 2 * 2^104
        const SECOND_LARGEST_F32: f32 = 340282326356119256160033759537265639424.;
        test!(SECOND_LARGEST_F32, f32 -> u128, 0xfffffe00000000000000000000000000);
    }
}
