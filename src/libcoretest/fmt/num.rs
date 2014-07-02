// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(unsigned_negate)]

use core::fmt::radix;

#[test]
fn test_format_int() {
    // Formatting integers should select the right implementation based off
    // the type of the argument. Also, hex/octal/binary should be defined
    // for integers, but they shouldn't emit the negative sign.
    assert!(format!("{}", 1i).as_slice() == "1");
    assert!(format!("{}", 1i8).as_slice() == "1");
    assert!(format!("{}", 1i16).as_slice() == "1");
    assert!(format!("{}", 1i32).as_slice() == "1");
    assert!(format!("{}", 1i64).as_slice() == "1");
    assert!(format!("{:d}", -1i).as_slice() == "-1");
    assert!(format!("{:d}", -1i8).as_slice() == "-1");
    assert!(format!("{:d}", -1i16).as_slice() == "-1");
    assert!(format!("{:d}", -1i32).as_slice() == "-1");
    assert!(format!("{:d}", -1i64).as_slice() == "-1");
    assert!(format!("{:t}", 1i).as_slice() == "1");
    assert!(format!("{:t}", 1i8).as_slice() == "1");
    assert!(format!("{:t}", 1i16).as_slice() == "1");
    assert!(format!("{:t}", 1i32).as_slice() == "1");
    assert!(format!("{:t}", 1i64).as_slice() == "1");
    assert!(format!("{:x}", 1i).as_slice() == "1");
    assert!(format!("{:x}", 1i8).as_slice() == "1");
    assert!(format!("{:x}", 1i16).as_slice() == "1");
    assert!(format!("{:x}", 1i32).as_slice() == "1");
    assert!(format!("{:x}", 1i64).as_slice() == "1");
    assert!(format!("{:X}", 1i).as_slice() == "1");
    assert!(format!("{:X}", 1i8).as_slice() == "1");
    assert!(format!("{:X}", 1i16).as_slice() == "1");
    assert!(format!("{:X}", 1i32).as_slice() == "1");
    assert!(format!("{:X}", 1i64).as_slice() == "1");
    assert!(format!("{:o}", 1i).as_slice() == "1");
    assert!(format!("{:o}", 1i8).as_slice() == "1");
    assert!(format!("{:o}", 1i16).as_slice() == "1");
    assert!(format!("{:o}", 1i32).as_slice() == "1");
    assert!(format!("{:o}", 1i64).as_slice() == "1");

    assert!(format!("{}", 1u).as_slice() == "1");
    assert!(format!("{}", 1u8).as_slice() == "1");
    assert!(format!("{}", 1u16).as_slice() == "1");
    assert!(format!("{}", 1u32).as_slice() == "1");
    assert!(format!("{}", 1u64).as_slice() == "1");
    assert!(format!("{:u}", 1u).as_slice() == "1");
    assert!(format!("{:u}", 1u8).as_slice() == "1");
    assert!(format!("{:u}", 1u16).as_slice() == "1");
    assert!(format!("{:u}", 1u32).as_slice() == "1");
    assert!(format!("{:u}", 1u64).as_slice() == "1");
    assert!(format!("{:t}", 1u).as_slice() == "1");
    assert!(format!("{:t}", 1u8).as_slice() == "1");
    assert!(format!("{:t}", 1u16).as_slice() == "1");
    assert!(format!("{:t}", 1u32).as_slice() == "1");
    assert!(format!("{:t}", 1u64).as_slice() == "1");
    assert!(format!("{:x}", 1u).as_slice() == "1");
    assert!(format!("{:x}", 1u8).as_slice() == "1");
    assert!(format!("{:x}", 1u16).as_slice() == "1");
    assert!(format!("{:x}", 1u32).as_slice() == "1");
    assert!(format!("{:x}", 1u64).as_slice() == "1");
    assert!(format!("{:X}", 1u).as_slice() == "1");
    assert!(format!("{:X}", 1u8).as_slice() == "1");
    assert!(format!("{:X}", 1u16).as_slice() == "1");
    assert!(format!("{:X}", 1u32).as_slice() == "1");
    assert!(format!("{:X}", 1u64).as_slice() == "1");
    assert!(format!("{:o}", 1u).as_slice() == "1");
    assert!(format!("{:o}", 1u8).as_slice() == "1");
    assert!(format!("{:o}", 1u16).as_slice() == "1");
    assert!(format!("{:o}", 1u32).as_slice() == "1");
    assert!(format!("{:o}", 1u64).as_slice() == "1");

    // Test a larger number
    assert!(format!("{:t}", 55i).as_slice() == "110111");
    assert!(format!("{:o}", 55i).as_slice() == "67");
    assert!(format!("{:d}", 55i).as_slice() == "55");
    assert!(format!("{:x}", 55i).as_slice() == "37");
    assert!(format!("{:X}", 55i).as_slice() == "37");
}

#[test]
fn test_format_int_zero() {
    assert!(format!("{}", 0i).as_slice() == "0");
    assert!(format!("{:d}", 0i).as_slice() == "0");
    assert!(format!("{:t}", 0i).as_slice() == "0");
    assert!(format!("{:o}", 0i).as_slice() == "0");
    assert!(format!("{:x}", 0i).as_slice() == "0");
    assert!(format!("{:X}", 0i).as_slice() == "0");

    assert!(format!("{}", 0u).as_slice() == "0");
    assert!(format!("{:u}", 0u).as_slice() == "0");
    assert!(format!("{:t}", 0u).as_slice() == "0");
    assert!(format!("{:o}", 0u).as_slice() == "0");
    assert!(format!("{:x}", 0u).as_slice() == "0");
    assert!(format!("{:X}", 0u).as_slice() == "0");
}

#[test]
fn test_format_int_flags() {
    assert!(format!("{:3d}", 1i).as_slice() == "  1");
    assert!(format!("{:>3d}", 1i).as_slice() == "  1");
    assert!(format!("{:>+3d}", 1i).as_slice() == " +1");
    assert!(format!("{:<3d}", 1i).as_slice() == "1  ");
    assert!(format!("{:#d}", 1i).as_slice() == "1");
    assert!(format!("{:#x}", 10i).as_slice() == "0xa");
    assert!(format!("{:#X}", 10i).as_slice() == "0xA");
    assert!(format!("{:#5x}", 10i).as_slice() == "  0xa");
    assert!(format!("{:#o}", 10i).as_slice() == "0o12");
    assert!(format!("{:08x}", 10i).as_slice() == "0000000a");
    assert!(format!("{:8x}", 10i).as_slice() == "       a");
    assert!(format!("{:<8x}", 10i).as_slice() == "a       ");
    assert!(format!("{:>8x}", 10i).as_slice() == "       a");
    assert!(format!("{:#08x}", 10i).as_slice() == "0x00000a");
    assert!(format!("{:08d}", -10i).as_slice() == "-0000010");
    assert!(format!("{:x}", -1u8).as_slice() == "ff");
    assert!(format!("{:X}", -1u8).as_slice() == "FF");
    assert!(format!("{:t}", -1u8).as_slice() == "11111111");
    assert!(format!("{:o}", -1u8).as_slice() == "377");
    assert!(format!("{:#x}", -1u8).as_slice() == "0xff");
    assert!(format!("{:#X}", -1u8).as_slice() == "0xFF");
    assert!(format!("{:#t}", -1u8).as_slice() == "0b11111111");
    assert!(format!("{:#o}", -1u8).as_slice() == "0o377");
}

#[test]
fn test_format_int_sign_padding() {
    assert!(format!("{:+5d}", 1i).as_slice() == "   +1");
    assert!(format!("{:+5d}", -1i).as_slice() == "   -1");
    assert!(format!("{:05d}", 1i).as_slice() == "00001");
    assert!(format!("{:05d}", -1i).as_slice() == "-0001");
    assert!(format!("{:+05d}", 1i).as_slice() == "+0001");
    assert!(format!("{:+05d}", -1i).as_slice() == "-0001");
}

#[test]
fn test_format_int_twos_complement() {
    use core::{i8, i16, i32, i64};
    assert!(format!("{}", i8::MIN).as_slice() == "-128");
    assert!(format!("{}", i16::MIN).as_slice() == "-32768");
    assert!(format!("{}", i32::MIN).as_slice() == "-2147483648");
    assert!(format!("{}", i64::MIN).as_slice() == "-9223372036854775808");
}

#[test]
fn test_format_radix() {
    assert!(format!("{:04}", radix(3i, 2)).as_slice() == "0011");
    assert!(format!("{}", radix(55i, 36)).as_slice() == "1j");
}

#[test]
#[should_fail]
fn test_radix_base_too_large() {
    let _ = radix(55i, 37);
}

mod uint {
    use test::Bencher;
    use core::fmt::radix;
    use std::rand::{weak_rng, Rng};

    #[bench]
    fn format_bin(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:t}", rng.gen::<uint>()); })
    }

    #[bench]
    fn format_oct(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:o}", rng.gen::<uint>()); })
    }

    #[bench]
    fn format_dec(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:u}", rng.gen::<uint>()); })
    }

    #[bench]
    fn format_hex(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:x}", rng.gen::<uint>()); })
    }

    #[bench]
    fn format_base_36(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{}", radix(rng.gen::<uint>(), 36)); })
    }
}

mod int {
    use test::Bencher;
    use core::fmt::radix;
    use std::rand::{weak_rng, Rng};

    #[bench]
    fn format_bin(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:t}", rng.gen::<int>()); })
    }

    #[bench]
    fn format_oct(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:o}", rng.gen::<int>()); })
    }

    #[bench]
    fn format_dec(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:d}", rng.gen::<int>()); })
    }

    #[bench]
    fn format_hex(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{:x}", rng.gen::<int>()); })
    }

    #[bench]
    fn format_base_36(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| { format!("{}", radix(rng.gen::<int>(), 36)); })
    }
}
