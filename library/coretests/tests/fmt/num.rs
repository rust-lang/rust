use std::mem::size_of;

#[test]
fn test_format_int_zero() {
    assert_eq!(format!("{}", 0), "0");
    assert_eq!(format!("{:?}", 0), "0");
    assert_eq!(format!("{:b}", 0), "0");
    assert_eq!(format!("{:o}", 0), "0");
    assert_eq!(format!("{:x}", 0), "0");
    assert_eq!(format!("{:X}", 0), "0");
    assert_eq!(format!("{:e}", 0), "0e0");
    assert_eq!(format!("{:E}", 0), "0E0");

    // again, with unsigned
    assert_eq!(format!("{}", 0u32), "0");
    assert_eq!(format!("{:?}", 0u32), "0");
    assert_eq!(format!("{:b}", 0u32), "0");
    assert_eq!(format!("{:o}", 0u32), "0");
    assert_eq!(format!("{:x}", 0u32), "0");
    assert_eq!(format!("{:X}", 0u32), "0");
    assert_eq!(format!("{:e}", 0u32), "0e0");
    assert_eq!(format!("{:E}", 0u32), "0E0");
}

#[test]
fn test_format_int_one() {
    // Formatting integers should select the right implementation based off
    // the type of the argument. Also, hex/octal/binary should be defined
    // for integers, but they shouldn't emit the negative sign.
    assert_eq!(format!("{}", 1isize), "1");
    assert_eq!(format!("{}", 1i8), "1");
    assert_eq!(format!("{}", 1i16), "1");
    assert_eq!(format!("{}", 1i32), "1");
    assert_eq!(format!("{}", 1i64), "1");
    assert_eq!(format!("{}", 1i128), "1");
    assert_eq!(format!("{:?}", 1isize), "1");
    assert_eq!(format!("{:?}", 1i8), "1");
    assert_eq!(format!("{:?}", 1i16), "1");
    assert_eq!(format!("{:?}", 1i32), "1");
    assert_eq!(format!("{:?}", 1i64), "1");
    assert_eq!(format!("{:?}", 1i128), "1");
    assert_eq!(format!("{:b}", 1isize), "1");
    assert_eq!(format!("{:b}", 1i8), "1");
    assert_eq!(format!("{:b}", 1i16), "1");
    assert_eq!(format!("{:b}", 1i32), "1");
    assert_eq!(format!("{:b}", 1i64), "1");
    assert_eq!(format!("{:b}", 1i128), "1");
    assert_eq!(format!("{:x}", 1isize), "1");
    assert_eq!(format!("{:x}", 1i8), "1");
    assert_eq!(format!("{:x}", 1i16), "1");
    assert_eq!(format!("{:x}", 1i32), "1");
    assert_eq!(format!("{:x}", 1i64), "1");
    assert_eq!(format!("{:x}", 1i128), "1");
    assert_eq!(format!("{:X}", 1isize), "1");
    assert_eq!(format!("{:X}", 1i8), "1");
    assert_eq!(format!("{:X}", 1i16), "1");
    assert_eq!(format!("{:X}", 1i32), "1");
    assert_eq!(format!("{:X}", 1i64), "1");
    assert_eq!(format!("{:X}", 1i128), "1");
    assert_eq!(format!("{:o}", 1isize), "1");
    assert_eq!(format!("{:o}", 1i8), "1");
    assert_eq!(format!("{:o}", 1i16), "1");
    assert_eq!(format!("{:o}", 1i32), "1");
    assert_eq!(format!("{:o}", 1i64), "1");
    assert_eq!(format!("{:o}", 1i128), "1");
    assert_eq!(format!("{:e}", 1isize), "1e0");
    assert_eq!(format!("{:e}", 1i8), "1e0");
    assert_eq!(format!("{:e}", 1i16), "1e0");
    assert_eq!(format!("{:e}", 1i32), "1e0");
    assert_eq!(format!("{:e}", 1i64), "1e0");
    assert_eq!(format!("{:e}", 1i128), "1e0");
    assert_eq!(format!("{:E}", 1isize), "1E0");
    assert_eq!(format!("{:E}", 1i8), "1E0");
    assert_eq!(format!("{:E}", 1i16), "1E0");
    assert_eq!(format!("{:E}", 1i32), "1E0");
    assert_eq!(format!("{:E}", 1i64), "1E0");
    assert_eq!(format!("{:E}", 1i128), "1E0");

    // again, with unsigned
    assert_eq!(format!("{}", 1usize), "1");
    assert_eq!(format!("{}", 1u8), "1");
    assert_eq!(format!("{}", 1u16), "1");
    assert_eq!(format!("{}", 1u32), "1");
    assert_eq!(format!("{}", 1u64), "1");
    assert_eq!(format!("{}", 1u128), "1");
    assert_eq!(format!("{:?}", 1usize), "1");
    assert_eq!(format!("{:?}", 1u8), "1");
    assert_eq!(format!("{:?}", 1u16), "1");
    assert_eq!(format!("{:?}", 1u32), "1");
    assert_eq!(format!("{:?}", 1u64), "1");
    assert_eq!(format!("{:?}", 1u128), "1");
    assert_eq!(format!("{:b}", 1usize), "1");
    assert_eq!(format!("{:b}", 1u8), "1");
    assert_eq!(format!("{:b}", 1u16), "1");
    assert_eq!(format!("{:b}", 1u32), "1");
    assert_eq!(format!("{:b}", 1u64), "1");
    assert_eq!(format!("{:b}", 1u128), "1");
    assert_eq!(format!("{:x}", 1usize), "1");
    assert_eq!(format!("{:x}", 1u8), "1");
    assert_eq!(format!("{:x}", 1u16), "1");
    assert_eq!(format!("{:x}", 1u32), "1");
    assert_eq!(format!("{:x}", 1u64), "1");
    assert_eq!(format!("{:x}", 1u128), "1");
    assert_eq!(format!("{:X}", 1usize), "1");
    assert_eq!(format!("{:X}", 1u8), "1");
    assert_eq!(format!("{:X}", 1u16), "1");
    assert_eq!(format!("{:X}", 1u32), "1");
    assert_eq!(format!("{:X}", 1u64), "1");
    assert_eq!(format!("{:X}", 1u128), "1");
    assert_eq!(format!("{:o}", 1usize), "1");
    assert_eq!(format!("{:o}", 1u8), "1");
    assert_eq!(format!("{:o}", 1u16), "1");
    assert_eq!(format!("{:o}", 1u32), "1");
    assert_eq!(format!("{:o}", 1u64), "1");
    assert_eq!(format!("{:o}", 1u128), "1");
    assert_eq!(format!("{:e}", 1u8), "1e0");
    assert_eq!(format!("{:e}", 1u16), "1e0");
    assert_eq!(format!("{:e}", 1u32), "1e0");
    assert_eq!(format!("{:e}", 1u64), "1e0");
    assert_eq!(format!("{:e}", 1u128), "1e0");
    assert_eq!(format!("{:E}", 1u8), "1E0");
    assert_eq!(format!("{:E}", 1u16), "1E0");
    assert_eq!(format!("{:E}", 1u32), "1E0");
    assert_eq!(format!("{:E}", 1u64), "1E0");
    assert_eq!(format!("{:E}", 1u128), "1E0");

    // again, with negative
    assert_eq!(format!("{}", -1isize), "-1");
    assert_eq!(format!("{}", -1i8), "-1");
    assert_eq!(format!("{}", -1i16), "-1");
    assert_eq!(format!("{}", -1i32), "-1");
    assert_eq!(format!("{}", -1i64), "-1");
    assert_eq!(format!("{}", -1i128), "-1");
    assert_eq!(format!("{:?}", -1isize), "-1");
    assert_eq!(format!("{:?}", -1i8), "-1");
    assert_eq!(format!("{:?}", -1i16), "-1");
    assert_eq!(format!("{:?}", -1i32), "-1");
    assert_eq!(format!("{:?}", -1i64), "-1");
    assert_eq!(format!("{:?}", -1i128), "-1");
    assert_eq!(format!("{:b}", -1isize), "1".repeat(size_of::<isize>() * 8));
    assert_eq!(format!("{:b}", -1i8), "11111111");
    assert_eq!(format!("{:b}", -1i16), "1111111111111111");
    assert_eq!(format!("{:b}", -1i32), "11111111111111111111111111111111");
    assert_eq!(
        format!("{:b}", -1i64),
        "1111111111111111111111111111111111111111111111111111111111111111"
    );
    assert_eq!(
        format!("{:b}", -1i128),
        "11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
    );
    assert_eq!(format!("{:x}", -1isize), "ff".repeat(size_of::<isize>()));
    assert_eq!(format!("{:x}", -1i8), "ff");
    assert_eq!(format!("{:x}", -1i16), "ffff");
    assert_eq!(format!("{:x}", -1i32), "ffffffff");
    assert_eq!(format!("{:x}", -1i64), "ffffffffffffffff");
    assert_eq!(format!("{:x}", -1i128), "ffffffffffffffffffffffffffffffff");
    assert_eq!(format!("{:X}", -1isize), "FF".repeat(size_of::<isize>()));
    assert_eq!(format!("{:X}", -1i8), "FF");
    assert_eq!(format!("{:X}", -1i16), "FFFF");
    assert_eq!(format!("{:X}", -1i32), "FFFFFFFF");
    assert_eq!(format!("{:X}", -1i64), "FFFFFFFFFFFFFFFF");
    assert_eq!(format!("{:X}", -1i128), "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    // octal test for isize omitted
    assert_eq!(format!("{:o}", -1i8), "377");
    assert_eq!(format!("{:o}", -1i16), "177777");
    assert_eq!(format!("{:o}", -1i32), "37777777777");
    assert_eq!(format!("{:o}", -1i64), "1777777777777777777777");
    assert_eq!(format!("{:o}", -1i128), "3777777777777777777777777777777777777777777");
    assert_eq!(format!("{:e}", -1isize), "-1e0");
    assert_eq!(format!("{:e}", -1i8), "-1e0");
    assert_eq!(format!("{:e}", -1i16), "-1e0");
    assert_eq!(format!("{:e}", -1i32), "-1e0");
    assert_eq!(format!("{:e}", -1i64), "-1e0");
    assert_eq!(format!("{:e}", -1i128), "-1e0");
    assert_eq!(format!("{:E}", -1isize), "-1E0");
    assert_eq!(format!("{:E}", -1i8), "-1E0");
    assert_eq!(format!("{:E}", -1i16), "-1E0");
    assert_eq!(format!("{:E}", -1i32), "-1E0");
    assert_eq!(format!("{:E}", -1i64), "-1E0");
    assert_eq!(format!("{:E}", -1i128), "-1E0");
}

#[test]
fn test_format_int_misc() {
    assert_eq!(format!("{:b}", 55), "110111");
    assert_eq!(format!("{:o}", 55), "67");
    assert_eq!(format!("{}", 55), "55");
    assert_eq!(format!("{:x}", 55), "37");
    assert_eq!(format!("{:X}", 55), "37");
    assert_eq!(format!("{:e}", 55), "5.5e1");
    assert_eq!(format!("{:E}", 55), "5.5E1");
    assert_eq!(format!("{:e}", 10000000000u64), "1e10");
    assert_eq!(format!("{:E}", 10000000000u64), "1E10");
    assert_eq!(format!("{:e}", 10000000001u64), "1.0000000001e10");
    assert_eq!(format!("{:E}", 10000000001u64), "1.0000000001E10");
}

#[test]
fn test_format_int_limits() {
    assert_eq!(format!("{}", i8::MIN), "-128");
    assert_eq!(format!("{}", i8::MAX), "127");
    assert_eq!(format!("{}", i16::MIN), "-32768");
    assert_eq!(format!("{}", i16::MAX), "32767");
    assert_eq!(format!("{}", i32::MIN), "-2147483648");
    assert_eq!(format!("{}", i32::MAX), "2147483647");
    assert_eq!(format!("{}", i64::MIN), "-9223372036854775808");
    assert_eq!(format!("{}", i64::MAX), "9223372036854775807");
    assert_eq!(format!("{}", i128::MIN), "-170141183460469231731687303715884105728");
    assert_eq!(format!("{}", i128::MAX), "170141183460469231731687303715884105727");

    assert_eq!(format!("{}", u8::MAX), "255");
    assert_eq!(format!("{}", u16::MAX), "65535");
    assert_eq!(format!("{}", u32::MAX), "4294967295");
    assert_eq!(format!("{}", u64::MAX), "18446744073709551615");
    assert_eq!(format!("{}", u128::MAX), "340282366920938463463374607431768211455");
}

#[test]
fn test_format_int_exp_limits() {
    assert_eq!(format!("{:e}", i8::MIN), "-1.28e2");
    assert_eq!(format!("{:e}", i8::MAX), "1.27e2");
    assert_eq!(format!("{:e}", i16::MIN), "-3.2768e4");
    assert_eq!(format!("{:e}", i16::MAX), "3.2767e4");
    assert_eq!(format!("{:e}", i32::MIN), "-2.147483648e9");
    assert_eq!(format!("{:e}", i32::MAX), "2.147483647e9");
    assert_eq!(format!("{:e}", i64::MIN), "-9.223372036854775808e18");
    assert_eq!(format!("{:e}", i64::MAX), "9.223372036854775807e18");
    assert_eq!(format!("{:e}", i128::MIN), "-1.70141183460469231731687303715884105728e38");
    assert_eq!(format!("{:e}", i128::MAX), "1.70141183460469231731687303715884105727e38");

    assert_eq!(format!("{:e}", u8::MAX), "2.55e2");
    assert_eq!(format!("{:e}", u16::MAX), "6.5535e4");
    assert_eq!(format!("{:e}", u32::MAX), "4.294967295e9");
    assert_eq!(format!("{:e}", u64::MAX), "1.8446744073709551615e19");
    assert_eq!(format!("{:e}", u128::MAX), "3.40282366920938463463374607431768211455e38");
}

#[test]
fn test_format_int_exp_precision() {
    //test that float and integer match
    let big_int: u32 = 314_159_265;
    assert_eq!(format!("{big_int:.1e}"), format!("{:.1e}", f64::from(big_int)));

    // test adding precision
    assert_eq!(format!("{:.10e}", i8::MIN), "-1.2800000000e2");
    assert_eq!(format!("{:.10e}", i16::MIN), "-3.2768000000e4");
    assert_eq!(format!("{:.10e}", i32::MIN), "-2.1474836480e9");
    assert_eq!(format!("{:.20e}", i64::MIN), "-9.22337203685477580800e18");
    assert_eq!(format!("{:.40e}", i128::MIN), "-1.7014118346046923173168730371588410572800e38");

    // test rounding
    assert_eq!(format!("{:.1e}", i8::MIN), "-1.3e2");
    assert_eq!(format!("{:.1e}", i16::MIN), "-3.3e4");
    assert_eq!(format!("{:.1e}", i32::MIN), "-2.1e9");
    assert_eq!(format!("{:.1e}", i64::MIN), "-9.2e18");
    assert_eq!(format!("{:.1e}", i128::MIN), "-1.7e38");

    // test huge precision
    assert_eq!(format!("{:.1000e}", 1), format!("1.{}e0", "0".repeat(1000)));
    //test zero precision
    assert_eq!(format!("{:.0e}", 1), format!("1e0",));
    assert_eq!(format!("{:.0e}", 35), format!("4e1",));

    // test padding with precision (and sign)
    assert_eq!(format!("{:+10.3e}", 1), "  +1.000e0");

    // test precision remains correct when rounding to next power
    #[cfg(miri)] // can't cover all of `i16` in Miri
    let range = [i16::MIN, -1, 1, i16::MAX];
    #[cfg(not(miri))]
    let range = i16::MIN..=i16::MAX;
    for i in range {
        for p in 0..=5 {
            assert_eq!(
                format!("{i:.p$e}"),
                format!("{:.p$e}", f32::from(i)),
                "integer {i} at precision {p}"
            );
        }
    }
}

#[test]
fn test_format_int_flags() {
    assert_eq!(format!("{:3}", 1), "  1");
    assert_eq!(format!("{:>3}", 1), "  1");
    assert_eq!(format!("{:>+3}", 1), " +1");
    assert_eq!(format!("{:<3}", 1), "1  ");
    assert_eq!(format!("{:#}", 1), "1");
    assert_eq!(format!("{:#x}", 10), "0xa");
    assert_eq!(format!("{:#X}", 10), "0xA");
    assert_eq!(format!("{:#5x}", 10), "  0xa");
    assert_eq!(format!("{:#o}", 10), "0o12");
    assert_eq!(format!("{:08x}", 10), "0000000a");
    assert_eq!(format!("{:8x}", 10), "       a");
    assert_eq!(format!("{:<8x}", 10), "a       ");
    assert_eq!(format!("{:>8x}", 10), "       a");
    assert_eq!(format!("{:#08x}", 10), "0x00000a");
    assert_eq!(format!("{:08}", -10), "-0000010");
    assert_eq!(format!("{:x}", !0u8), "ff");
    assert_eq!(format!("{:X}", !0u8), "FF");
    assert_eq!(format!("{:b}", !0u8), "11111111");
    assert_eq!(format!("{:o}", !0u8), "377");
    assert_eq!(format!("{:#x}", !0u8), "0xff");
    assert_eq!(format!("{:#X}", !0u8), "0xFF");
    assert_eq!(format!("{:#b}", !0u8), "0b11111111");
    assert_eq!(format!("{:#o}", !0u8), "0o377");
}

#[test]
fn test_format_int_sign_padding() {
    assert_eq!(format!("{:+5}", 1), "   +1");
    assert_eq!(format!("{:+5}", -1), "   -1");
    assert_eq!(format!("{:05}", 1), "00001");
    assert_eq!(format!("{:05}", -1), "-0001");
    assert_eq!(format!("{:+05}", 1), "+0001");
    assert_eq!(format!("{:+05}", -1), "-0001");
}

#[test]
fn test_format_debug_hex() {
    assert_eq!(format!("{:02x?}", b"Foo\0"), "[46, 6f, 6f, 00]");
    assert_eq!(format!("{:02X?}", b"Foo\0"), "[46, 6F, 6F, 00]");
}

#[test]
#[should_panic = "Formatting argument out of range"]
fn test_rt_width_too_long() {
    let _ = format!("Hello {:width$}!", "x", width = u16::MAX as usize + 1);
}
