#[test]
fn test_format_int() {
    // Formatting integers should select the right implementation based off
    // the type of the argument. Also, hex/octal/binary should be defined
    // for integers, but they shouldn't emit the negative sign.
    assert_eq!(format!("{}", 1isize), "1");
    assert_eq!(format!("{}", 1i8), "1");
    assert_eq!(format!("{}", 1i16), "1");
    assert_eq!(format!("{}", 1i32), "1");
    assert_eq!(format!("{}", 1i64), "1");
    assert_eq!(format!("{}", -1isize), "-1");
    assert_eq!(format!("{}", -1i8), "-1");
    assert_eq!(format!("{}", -1i16), "-1");
    assert_eq!(format!("{}", -1i32), "-1");
    assert_eq!(format!("{}", -1i64), "-1");
    assert_eq!(format!("{:?}", 1isize), "1");
    assert_eq!(format!("{:?}", 1i8), "1");
    assert_eq!(format!("{:?}", 1i16), "1");
    assert_eq!(format!("{:?}", 1i32), "1");
    assert_eq!(format!("{:?}", 1i64), "1");
    assert_eq!(format!("{:b}", 1isize), "1");
    assert_eq!(format!("{:b}", 1i8), "1");
    assert_eq!(format!("{:b}", 1i16), "1");
    assert_eq!(format!("{:b}", 1i32), "1");
    assert_eq!(format!("{:b}", 1i64), "1");
    assert_eq!(format!("{:x}", 1isize), "1");
    assert_eq!(format!("{:x}", 1i8), "1");
    assert_eq!(format!("{:x}", 1i16), "1");
    assert_eq!(format!("{:x}", 1i32), "1");
    assert_eq!(format!("{:x}", 1i64), "1");
    assert_eq!(format!("{:X}", 1isize), "1");
    assert_eq!(format!("{:X}", 1i8), "1");
    assert_eq!(format!("{:X}", 1i16), "1");
    assert_eq!(format!("{:X}", 1i32), "1");
    assert_eq!(format!("{:X}", 1i64), "1");
    assert_eq!(format!("{:o}", 1isize), "1");
    assert_eq!(format!("{:o}", 1i8), "1");
    assert_eq!(format!("{:o}", 1i16), "1");
    assert_eq!(format!("{:o}", 1i32), "1");
    assert_eq!(format!("{:o}", 1i64), "1");

    assert_eq!(format!("{}", 1usize), "1");
    assert_eq!(format!("{}", 1u8), "1");
    assert_eq!(format!("{}", 1u16), "1");
    assert_eq!(format!("{}", 1u32), "1");
    assert_eq!(format!("{}", 1u64), "1");
    assert_eq!(format!("{:?}", 1usize), "1");
    assert_eq!(format!("{:?}", 1u8), "1");
    assert_eq!(format!("{:?}", 1u16), "1");
    assert_eq!(format!("{:?}", 1u32), "1");
    assert_eq!(format!("{:?}", 1u64), "1");
    assert_eq!(format!("{:b}", 1usize), "1");
    assert_eq!(format!("{:b}", 1u8), "1");
    assert_eq!(format!("{:b}", 1u16), "1");
    assert_eq!(format!("{:b}", 1u32), "1");
    assert_eq!(format!("{:b}", 1u64), "1");
    assert_eq!(format!("{:x}", 1usize), "1");
    assert_eq!(format!("{:x}", 1u8), "1");
    assert_eq!(format!("{:x}", 1u16), "1");
    assert_eq!(format!("{:x}", 1u32), "1");
    assert_eq!(format!("{:x}", 1u64), "1");
    assert_eq!(format!("{:X}", 1usize), "1");
    assert_eq!(format!("{:X}", 1u8), "1");
    assert_eq!(format!("{:X}", 1u16), "1");
    assert_eq!(format!("{:X}", 1u32), "1");
    assert_eq!(format!("{:X}", 1u64), "1");
    assert_eq!(format!("{:o}", 1usize), "1");
    assert_eq!(format!("{:o}", 1u8), "1");
    assert_eq!(format!("{:o}", 1u16), "1");
    assert_eq!(format!("{:o}", 1u32), "1");
    assert_eq!(format!("{:o}", 1u64), "1");

    // Test a larger number
    assert_eq!(format!("{:b}", 55), "110111");
    assert_eq!(format!("{:o}", 55), "67");
    assert_eq!(format!("{}", 55), "55");
    assert_eq!(format!("{:x}", 55), "37");
    assert_eq!(format!("{:X}", 55), "37");
}

#[test]
fn test_format_int_zero() {
    assert_eq!(format!("{}", 0), "0");
    assert_eq!(format!("{:?}", 0), "0");
    assert_eq!(format!("{:b}", 0), "0");
    assert_eq!(format!("{:o}", 0), "0");
    assert_eq!(format!("{:x}", 0), "0");
    assert_eq!(format!("{:X}", 0), "0");

    assert_eq!(format!("{}", 0u32), "0");
    assert_eq!(format!("{:?}", 0u32), "0");
    assert_eq!(format!("{:b}", 0u32), "0");
    assert_eq!(format!("{:o}", 0u32), "0");
    assert_eq!(format!("{:x}", 0u32), "0");
    assert_eq!(format!("{:X}", 0u32), "0");
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
fn test_format_int_twos_complement() {
    use core::{i16, i32, i64, i8};
    assert_eq!(format!("{}", i8::MIN), "-128");
    assert_eq!(format!("{}", i16::MIN), "-32768");
    assert_eq!(format!("{}", i32::MIN), "-2147483648");
    assert_eq!(format!("{}", i64::MIN), "-9223372036854775808");
}

#[test]
fn test_format_debug_hex() {
    assert_eq!(format!("{:02x?}", b"Foo\0"), "[46, 6f, 6f, 00]");
    assert_eq!(format!("{:02X?}", b"Foo\0"), "[46, 6F, 6F, 00]");
}
