// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(unsigned_negation)]

use core::fmt::radix;

#[test]
fn test_format_int() {
    // Formatting integers should select the right implementation based off
    // the type of the argument. Also, hex/octal/binary should be defined
    // for integers, but they shouldn't emit the negative sign.
    assert!(format!("{}", 1isize) == "1");
    assert!(format!("{}", 1i8) == "1");
    assert!(format!("{}", 1i16) == "1");
    assert!(format!("{}", 1i32) == "1");
    assert!(format!("{}", 1i64) == "1");
    assert!(format!("{}", -1isize) == "-1");
    assert!(format!("{}", -1i8) == "-1");
    assert!(format!("{}", -1i16) == "-1");
    assert!(format!("{}", -1i32) == "-1");
    assert!(format!("{}", -1i64) == "-1");
    assert!(format!("{:?}", 1isize) == "1");
    assert!(format!("{:?}", 1i8) == "1");
    assert!(format!("{:?}", 1i16) == "1");
    assert!(format!("{:?}", 1i32) == "1");
    assert!(format!("{:?}", 1i64) == "1");
    assert!(format!("{:b}", 1isize) == "1");
    assert!(format!("{:b}", 1i8) == "1");
    assert!(format!("{:b}", 1i16) == "1");
    assert!(format!("{:b}", 1i32) == "1");
    assert!(format!("{:b}", 1i64) == "1");
    assert!(format!("{:x}", 1isize) == "1");
    assert!(format!("{:x}", 1i8) == "1");
    assert!(format!("{:x}", 1i16) == "1");
    assert!(format!("{:x}", 1i32) == "1");
    assert!(format!("{:x}", 1i64) == "1");
    assert!(format!("{:X}", 1isize) == "1");
    assert!(format!("{:X}", 1i8) == "1");
    assert!(format!("{:X}", 1i16) == "1");
    assert!(format!("{:X}", 1i32) == "1");
    assert!(format!("{:X}", 1i64) == "1");
    assert!(format!("{:o}", 1isize) == "1");
    assert!(format!("{:o}", 1i8) == "1");
    assert!(format!("{:o}", 1i16) == "1");
    assert!(format!("{:o}", 1i32) == "1");
    assert!(format!("{:o}", 1i64) == "1");

    assert!(format!("{}", 1usize) == "1");
    assert!(format!("{}", 1u8) == "1");
    assert!(format!("{}", 1u16) == "1");
    assert!(format!("{}", 1u32) == "1");
    assert!(format!("{}", 1u64) == "1");
    assert!(format!("{:?}", 1usize) == "1");
    assert!(format!("{:?}", 1u8) == "1");
    assert!(format!("{:?}", 1u16) == "1");
    assert!(format!("{:?}", 1u32) == "1");
    assert!(format!("{:?}", 1u64) == "1");
    assert!(format!("{:b}", 1usize) == "1");
    assert!(format!("{:b}", 1u8) == "1");
    assert!(format!("{:b}", 1u16) == "1");
    assert!(format!("{:b}", 1u32) == "1");
    assert!(format!("{:b}", 1u64) == "1");
    assert!(format!("{:x}", 1usize) == "1");
    assert!(format!("{:x}", 1u8) == "1");
    assert!(format!("{:x}", 1u16) == "1");
    assert!(format!("{:x}", 1u32) == "1");
    assert!(format!("{:x}", 1u64) == "1");
    assert!(format!("{:X}", 1usize) == "1");
    assert!(format!("{:X}", 1u8) == "1");
    assert!(format!("{:X}", 1u16) == "1");
    assert!(format!("{:X}", 1u32) == "1");
    assert!(format!("{:X}", 1u64) == "1");
    assert!(format!("{:o}", 1usize) == "1");
    assert!(format!("{:o}", 1u8) == "1");
    assert!(format!("{:o}", 1u16) == "1");
    assert!(format!("{:o}", 1u32) == "1");
    assert!(format!("{:o}", 1u64) == "1");

    // Test a larger number
    assert!(format!("{:b}", 55) == "110111");
    assert!(format!("{:o}", 55) == "67");
    assert!(format!("{}", 55) == "55");
    assert!(format!("{:x}", 55) == "37");
    assert!(format!("{:X}", 55) == "37");
}

#[test]
fn test_format_int_zero() {
    assert!(format!("{}", 0) == "0");
    assert!(format!("{:?}", 0) == "0");
    assert!(format!("{:b}", 0) == "0");
    assert!(format!("{:o}", 0) == "0");
    assert!(format!("{:x}", 0) == "0");
    assert!(format!("{:X}", 0) == "0");

    assert!(format!("{}", 0u32) == "0");
    assert!(format!("{:?}", 0u32) == "0");
    assert!(format!("{:b}", 0u32) == "0");
    assert!(format!("{:o}", 0u32) == "0");
    assert!(format!("{:x}", 0u32) == "0");
    assert!(format!("{:X}", 0u32) == "0");
}

#[test]
fn test_format_int_flags() {
    assert!(format!("{:3}", 1) == "  1");
    assert!(format!("{:>3}", 1) == "  1");
    assert!(format!("{:>+3}", 1) == " +1");
    assert!(format!("{:<3}", 1) == "1  ");
    assert!(format!("{:#}", 1) == "1");
    assert!(format!("{:#x}", 10) == "0xa");
    assert!(format!("{:#X}", 10) == "0xA");
    assert!(format!("{:#5x}", 10) == "  0xa");
    assert!(format!("{:#o}", 10) == "0o12");
    assert!(format!("{:08x}", 10) == "0000000a");
    assert!(format!("{:8x}", 10) == "       a");
    assert!(format!("{:<8x}", 10) == "a       ");
    assert!(format!("{:>8x}", 10) == "       a");
    assert!(format!("{:#08x}", 10) == "0x00000a");
    assert!(format!("{:08}", -10) == "-0000010");
    assert!(format!("{:x}", -1u8) == "ff");
    assert!(format!("{:X}", -1u8) == "FF");
    assert!(format!("{:b}", -1u8) == "11111111");
    assert!(format!("{:o}", -1u8) == "377");
    assert!(format!("{:#x}", -1u8) == "0xff");
    assert!(format!("{:#X}", -1u8) == "0xFF");
    assert!(format!("{:#b}", -1u8) == "0b11111111");
    assert!(format!("{:#o}", -1u8) == "0o377");
}

#[test]
fn test_format_int_sign_padding() {
    assert!(format!("{:+5}", 1) == "   +1");
    assert!(format!("{:+5}", -1) == "   -1");
    assert!(format!("{:05}", 1) == "00001");
    assert!(format!("{:05}", -1) == "-0001");
    assert!(format!("{:+05}", 1) == "+0001");
    assert!(format!("{:+05}", -1) == "-0001");
}

#[test]
fn test_format_int_twos_complement() {
    use core::{i8, i16, i32, i64};
    assert!(format!("{}", i8::MIN) == "-128");
    assert!(format!("{}", i16::MIN) == "-32768");
    assert!(format!("{}", i32::MIN) == "-2147483648");
    assert!(format!("{}", i64::MIN) == "-9223372036854775808");
}

#[test]
fn test_format_radix() {
    assert!(format!("{:04}", radix(3, 2)) == "0011");
    assert!(format!("{}", radix(55, 36)) == "1j");
}

#[test]
#[should_fail]
fn test_radix_base_too_large() {
    let _ = radix(55, 37);
}

mod u32 {
    use test::Bencher;
    use core::fmt::radix;
    use std::rand::{weak_rng, Rng};
    use std::old_io::util::NullWriter;

    #[bench]
    fn format_bin(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:b}", rng.gen::<u32>()) })
    }

    #[bench]
    fn format_oct(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:o}", rng.gen::<u32>()) })
    }

    #[bench]
    fn format_dec(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{}", rng.gen::<u32>()) })
    }

    #[bench]
    fn format_hex(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:x}", rng.gen::<u32>()) })
    }

    #[bench]
    fn format_show(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:?}", rng.gen::<u32>()) })
    }

    #[bench]
    fn format_base_36(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{}", radix(rng.gen::<u32>(), 36)) })
    }
}

mod i32 {
    use test::Bencher;
    use core::fmt::radix;
    use std::rand::{weak_rng, Rng};
    use std::old_io::util::NullWriter;

    #[bench]
    fn format_bin(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:b}", rng.gen::<i32>()) })
    }

    #[bench]
    fn format_oct(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:o}", rng.gen::<i32>()) })
    }

    #[bench]
    fn format_dec(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{}", rng.gen::<i32>()) })
    }

    #[bench]
    fn format_hex(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:x}", rng.gen::<i32>()) })
    }

    #[bench]
    fn format_show(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{:?}", rng.gen::<i32>()) })
    }

    #[bench]
    fn format_base_36(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { write!(&mut NullWriter, "{}", radix(rng.gen::<i32>(), 36)) })
    }
}
