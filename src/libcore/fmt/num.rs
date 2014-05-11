// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integer and floating-point number formatting

// FIXME: #6220 Implement floating point formatting

#![allow(unsigned_negate)]

use container::Container;
use fmt;
use iter::{Iterator, DoubleEndedIterator};
use num::{Int, cast, zero};
use option::{Some, None};
use slice::{ImmutableVector, MutableVector};

/// A type that represents a specific radix
trait GenericRadix {
    /// The number of digits.
    fn base(&self) -> u8;

    /// A radix-specific prefix string.
    fn prefix(&self) -> &'static str { "" }

    /// Converts an integer to corresponding radix digit.
    fn digit(&self, x: u8) -> u8;

    /// Format an integer using the radix using a formatter.
    fn fmt_int<T: Int>(&self, mut x: T, f: &mut fmt::Formatter) -> fmt::Result {
        // The radix can be as low as 2, so we need a buffer of at least 64
        // characters for a base 2 number.
        let mut buf = [0u8, ..64];
        let base = cast(self.base()).unwrap();
        let mut curr = buf.len();
        let is_positive = x >= zero();
        if is_positive {
            // Accumulate each digit of the number from the least significant
            // to the most significant figure.
            for byte in buf.mut_iter().rev() {
                let n = x % base;                         // Get the current place value.
                x = x / base;                             // Deaccumulate the number.
                *byte = self.digit(cast(n).unwrap());     // Store the digit in the buffer.
                curr -= 1;
                if x == zero() { break; }                 // No more digits left to accumulate.
            }
        } else {
            // Do the same as above, but accounting for two's complement.
            for byte in buf.mut_iter().rev() {
                let n = -(x % base);                      // Get the current place value.
                x = x / base;                             // Deaccumulate the number.
                *byte = self.digit(cast(n).unwrap());     // Store the digit in the buffer.
                curr -= 1;
                if x == zero() { break; }                 // No more digits left to accumulate.
            }
        }
        f.pad_integral(is_positive, self.prefix(), buf.slice_from(curr))
    }
}

/// A binary (base 2) radix
#[deriving(Clone, Eq)]
struct Binary;

/// An octal (base 8) radix
#[deriving(Clone, Eq)]
struct Octal;

/// A decimal (base 10) radix
#[deriving(Clone, Eq)]
struct Decimal;

/// A hexadecimal (base 16) radix, formatted with lower-case characters
#[deriving(Clone, Eq)]
struct LowerHex;

/// A hexadecimal (base 16) radix, formatted with upper-case characters
#[deriving(Clone, Eq)]
pub struct UpperHex;

macro_rules! radix {
    ($T:ident, $base:expr, $prefix:expr, $($x:pat => $conv:expr),+) => {
        impl GenericRadix for $T {
            fn base(&self) -> u8 { $base }
            fn prefix(&self) -> &'static str { $prefix }
            fn digit(&self, x: u8) -> u8 {
                match x {
                    $($x => $conv,)+
                    x => fail!("number not in the range 0..{}: {}", self.base() - 1, x),
                }
            }
        }
    }
}

radix!(Binary,    2, "0b", x @  0 .. 2 => '0' as u8 + x)
radix!(Octal,     8, "0o", x @  0 .. 7 => '0' as u8 + x)
radix!(Decimal,  10, "",   x @  0 .. 9 => '0' as u8 + x)
radix!(LowerHex, 16, "0x", x @  0 .. 9 => '0' as u8 + x,
                           x @ 10 ..15 => 'a' as u8 + (x - 10))
radix!(UpperHex, 16, "0x", x @  0 .. 9 => '0' as u8 + x,
                           x @ 10 ..15 => 'A' as u8 + (x - 10))

/// A radix with in the range of `2..36`.
#[deriving(Clone, Eq)]
pub struct Radix {
    base: u8,
}

impl Radix {
    fn new(base: u8) -> Radix {
        assert!(2 <= base && base <= 36, "the base must be in the range of 0..36: {}", base);
        Radix { base: base }
    }
}

impl GenericRadix for Radix {
    fn base(&self) -> u8 { self.base }
    fn digit(&self, x: u8) -> u8 {
        match x {
            x @  0 ..9 => '0' as u8 + x,
            x if x < self.base() => 'a' as u8 + (x - 10),
            x => fail!("number not in the range 0..{}: {}", self.base() - 1, x),
        }
    }
}

/// A helper type for formatting radixes.
pub struct RadixFmt<T, R>(T, R);

/// Constructs a radix formatter in the range of `2..36`.
///
/// # Example
///
/// ~~~
/// use std::fmt::radix;
/// assert_eq!(format!("{}", radix(55, 36)), "1j".to_owned());
/// ~~~
pub fn radix<T>(x: T, base: u8) -> RadixFmt<T, Radix> {
    RadixFmt(x, Radix::new(base))
}

macro_rules! radix_fmt {
    ($T:ty as $U:ty, $fmt:ident) => {
        impl fmt::Show for RadixFmt<$T, Radix> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match *self { RadixFmt(ref x, radix) => radix.$fmt(*x as $U, f) }
            }
        }
    }
}
macro_rules! int_base {
    ($Trait:ident for $T:ident as $U:ident -> $Radix:ident) => {
        impl fmt::$Trait for $T {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                $Radix.fmt_int(*self as $U, f)
            }
        }
    }
}
macro_rules! integer {
    ($Int:ident, $Uint:ident) => {
        int_base!(Show     for $Int as $Int   -> Decimal)
        int_base!(Signed   for $Int as $Int   -> Decimal)
        int_base!(Binary   for $Int as $Uint  -> Binary)
        int_base!(Octal    for $Int as $Uint  -> Octal)
        int_base!(LowerHex for $Int as $Uint  -> LowerHex)
        int_base!(UpperHex for $Int as $Uint  -> UpperHex)
        radix_fmt!($Int as $Int, fmt_int)

        int_base!(Show     for $Uint as $Uint -> Decimal)
        int_base!(Unsigned for $Uint as $Uint -> Decimal)
        int_base!(Binary   for $Uint as $Uint -> Binary)
        int_base!(Octal    for $Uint as $Uint -> Octal)
        int_base!(LowerHex for $Uint as $Uint -> LowerHex)
        int_base!(UpperHex for $Uint as $Uint -> UpperHex)
        radix_fmt!($Uint as $Uint, fmt_int)
    }
}
integer!(int, uint)
integer!(i8, u8)
integer!(i16, u16)
integer!(i32, u32)
integer!(i64, u64)

#[cfg(test)]
mod tests {
    use fmt::radix;
    use super::{Binary, Octal, Decimal, LowerHex, UpperHex};
    use super::{GenericRadix, Radix};
    use realstd::str::StrAllocating;

    #[test]
    fn test_radix_base() {
        assert_eq!(Binary.base(), 2);
        assert_eq!(Octal.base(), 8);
        assert_eq!(Decimal.base(), 10);
        assert_eq!(LowerHex.base(), 16);
        assert_eq!(UpperHex.base(), 16);
        assert_eq!(Radix { base: 36 }.base(), 36);
    }

    #[test]
    fn test_radix_prefix() {
        assert_eq!(Binary.prefix(), "0b");
        assert_eq!(Octal.prefix(), "0o");
        assert_eq!(Decimal.prefix(), "");
        assert_eq!(LowerHex.prefix(), "0x");
        assert_eq!(UpperHex.prefix(), "0x");
        assert_eq!(Radix { base: 36 }.prefix(), "");
    }

    #[test]
    fn test_radix_digit() {
        assert_eq!(Binary.digit(0), '0' as u8);
        assert_eq!(Binary.digit(2), '2' as u8);
        assert_eq!(Octal.digit(0), '0' as u8);
        assert_eq!(Octal.digit(7), '7' as u8);
        assert_eq!(Decimal.digit(0), '0' as u8);
        assert_eq!(Decimal.digit(9), '9' as u8);
        assert_eq!(LowerHex.digit(0), '0' as u8);
        assert_eq!(LowerHex.digit(10), 'a' as u8);
        assert_eq!(LowerHex.digit(15), 'f' as u8);
        assert_eq!(UpperHex.digit(0), '0' as u8);
        assert_eq!(UpperHex.digit(10), 'A' as u8);
        assert_eq!(UpperHex.digit(15), 'F' as u8);
        assert_eq!(Radix { base: 36 }.digit(0), '0' as u8);
        assert_eq!(Radix { base: 36 }.digit(15), 'f' as u8);
        assert_eq!(Radix { base: 36 }.digit(35), 'z' as u8);
    }

    #[test]
    #[should_fail]
    fn test_hex_radix_digit_overflow() {
        let _ = LowerHex.digit(16);
    }

    #[test]
    fn test_format_int() {
        // Formatting integers should select the right implementation based off
        // the type of the argument. Also, hex/octal/binary should be defined
        // for integers, but they shouldn't emit the negative sign.
        assert_eq!(format!("{}", 1i), "1".to_owned());
        assert_eq!(format!("{}", 1i8), "1".to_owned());
        assert_eq!(format!("{}", 1i16), "1".to_owned());
        assert_eq!(format!("{}", 1i32), "1".to_owned());
        assert_eq!(format!("{}", 1i64), "1".to_owned());
        assert_eq!(format!("{:d}", -1i), "-1".to_owned());
        assert_eq!(format!("{:d}", -1i8), "-1".to_owned());
        assert_eq!(format!("{:d}", -1i16), "-1".to_owned());
        assert_eq!(format!("{:d}", -1i32), "-1".to_owned());
        assert_eq!(format!("{:d}", -1i64), "-1".to_owned());
        assert_eq!(format!("{:t}", 1i), "1".to_owned());
        assert_eq!(format!("{:t}", 1i8), "1".to_owned());
        assert_eq!(format!("{:t}", 1i16), "1".to_owned());
        assert_eq!(format!("{:t}", 1i32), "1".to_owned());
        assert_eq!(format!("{:t}", 1i64), "1".to_owned());
        assert_eq!(format!("{:x}", 1i), "1".to_owned());
        assert_eq!(format!("{:x}", 1i8), "1".to_owned());
        assert_eq!(format!("{:x}", 1i16), "1".to_owned());
        assert_eq!(format!("{:x}", 1i32), "1".to_owned());
        assert_eq!(format!("{:x}", 1i64), "1".to_owned());
        assert_eq!(format!("{:X}", 1i), "1".to_owned());
        assert_eq!(format!("{:X}", 1i8), "1".to_owned());
        assert_eq!(format!("{:X}", 1i16), "1".to_owned());
        assert_eq!(format!("{:X}", 1i32), "1".to_owned());
        assert_eq!(format!("{:X}", 1i64), "1".to_owned());
        assert_eq!(format!("{:o}", 1i), "1".to_owned());
        assert_eq!(format!("{:o}", 1i8), "1".to_owned());
        assert_eq!(format!("{:o}", 1i16), "1".to_owned());
        assert_eq!(format!("{:o}", 1i32), "1".to_owned());
        assert_eq!(format!("{:o}", 1i64), "1".to_owned());

        assert_eq!(format!("{}", 1u), "1".to_owned());
        assert_eq!(format!("{}", 1u8), "1".to_owned());
        assert_eq!(format!("{}", 1u16), "1".to_owned());
        assert_eq!(format!("{}", 1u32), "1".to_owned());
        assert_eq!(format!("{}", 1u64), "1".to_owned());
        assert_eq!(format!("{:u}", 1u), "1".to_owned());
        assert_eq!(format!("{:u}", 1u8), "1".to_owned());
        assert_eq!(format!("{:u}", 1u16), "1".to_owned());
        assert_eq!(format!("{:u}", 1u32), "1".to_owned());
        assert_eq!(format!("{:u}", 1u64), "1".to_owned());
        assert_eq!(format!("{:t}", 1u), "1".to_owned());
        assert_eq!(format!("{:t}", 1u8), "1".to_owned());
        assert_eq!(format!("{:t}", 1u16), "1".to_owned());
        assert_eq!(format!("{:t}", 1u32), "1".to_owned());
        assert_eq!(format!("{:t}", 1u64), "1".to_owned());
        assert_eq!(format!("{:x}", 1u), "1".to_owned());
        assert_eq!(format!("{:x}", 1u8), "1".to_owned());
        assert_eq!(format!("{:x}", 1u16), "1".to_owned());
        assert_eq!(format!("{:x}", 1u32), "1".to_owned());
        assert_eq!(format!("{:x}", 1u64), "1".to_owned());
        assert_eq!(format!("{:X}", 1u), "1".to_owned());
        assert_eq!(format!("{:X}", 1u8), "1".to_owned());
        assert_eq!(format!("{:X}", 1u16), "1".to_owned());
        assert_eq!(format!("{:X}", 1u32), "1".to_owned());
        assert_eq!(format!("{:X}", 1u64), "1".to_owned());
        assert_eq!(format!("{:o}", 1u), "1".to_owned());
        assert_eq!(format!("{:o}", 1u8), "1".to_owned());
        assert_eq!(format!("{:o}", 1u16), "1".to_owned());
        assert_eq!(format!("{:o}", 1u32), "1".to_owned());
        assert_eq!(format!("{:o}", 1u64), "1".to_owned());

        // Test a larger number
        assert_eq!(format!("{:t}", 55), "110111".to_owned());
        assert_eq!(format!("{:o}", 55), "67".to_owned());
        assert_eq!(format!("{:d}", 55), "55".to_owned());
        assert_eq!(format!("{:x}", 55), "37".to_owned());
        assert_eq!(format!("{:X}", 55), "37".to_owned());
    }

    #[test]
    fn test_format_int_zero() {
        assert_eq!(format!("{}", 0i), "0".to_owned());
        assert_eq!(format!("{:d}", 0i), "0".to_owned());
        assert_eq!(format!("{:t}", 0i), "0".to_owned());
        assert_eq!(format!("{:o}", 0i), "0".to_owned());
        assert_eq!(format!("{:x}", 0i), "0".to_owned());
        assert_eq!(format!("{:X}", 0i), "0".to_owned());

        assert_eq!(format!("{}", 0u), "0".to_owned());
        assert_eq!(format!("{:u}", 0u), "0".to_owned());
        assert_eq!(format!("{:t}", 0u), "0".to_owned());
        assert_eq!(format!("{:o}", 0u), "0".to_owned());
        assert_eq!(format!("{:x}", 0u), "0".to_owned());
        assert_eq!(format!("{:X}", 0u), "0".to_owned());
    }

    #[test]
    fn test_format_int_flags() {
        assert_eq!(format!("{:3d}", 1), "  1".to_owned());
        assert_eq!(format!("{:>3d}", 1), "  1".to_owned());
        assert_eq!(format!("{:>+3d}", 1), " +1".to_owned());
        assert_eq!(format!("{:<3d}", 1), "1  ".to_owned());
        assert_eq!(format!("{:#d}", 1), "1".to_owned());
        assert_eq!(format!("{:#x}", 10), "0xa".to_owned());
        assert_eq!(format!("{:#X}", 10), "0xA".to_owned());
        assert_eq!(format!("{:#5x}", 10), "  0xa".to_owned());
        assert_eq!(format!("{:#o}", 10), "0o12".to_owned());
        assert_eq!(format!("{:08x}", 10), "0000000a".to_owned());
        assert_eq!(format!("{:8x}", 10), "       a".to_owned());
        assert_eq!(format!("{:<8x}", 10), "a       ".to_owned());
        assert_eq!(format!("{:>8x}", 10), "       a".to_owned());
        assert_eq!(format!("{:#08x}", 10), "0x00000a".to_owned());
        assert_eq!(format!("{:08d}", -10), "-0000010".to_owned());
        assert_eq!(format!("{:x}", -1u8), "ff".to_owned());
        assert_eq!(format!("{:X}", -1u8), "FF".to_owned());
        assert_eq!(format!("{:t}", -1u8), "11111111".to_owned());
        assert_eq!(format!("{:o}", -1u8), "377".to_owned());
        assert_eq!(format!("{:#x}", -1u8), "0xff".to_owned());
        assert_eq!(format!("{:#X}", -1u8), "0xFF".to_owned());
        assert_eq!(format!("{:#t}", -1u8), "0b11111111".to_owned());
        assert_eq!(format!("{:#o}", -1u8), "0o377".to_owned());
    }

    #[test]
    fn test_format_int_sign_padding() {
        assert_eq!(format!("{:+5d}", 1), "   +1".to_owned());
        assert_eq!(format!("{:+5d}", -1), "   -1".to_owned());
        assert_eq!(format!("{:05d}", 1), "00001".to_owned());
        assert_eq!(format!("{:05d}", -1), "-0001".to_owned());
        assert_eq!(format!("{:+05d}", 1), "+0001".to_owned());
        assert_eq!(format!("{:+05d}", -1), "-0001".to_owned());
    }

    #[test]
    fn test_format_int_twos_complement() {
        use {i8, i16, i32, i64};
        assert_eq!(format!("{}", i8::MIN), "-128".to_owned());
        assert_eq!(format!("{}", i16::MIN), "-32768".to_owned());
        assert_eq!(format!("{}", i32::MIN), "-2147483648".to_owned());
        assert_eq!(format!("{}", i64::MIN), "-9223372036854775808".to_owned());
    }

    #[test]
    fn test_format_radix() {
        assert_eq!(format!("{:04}", radix(3, 2)), "0011".to_owned());
        assert_eq!(format!("{}", radix(55, 36)), "1j".to_owned());
    }

    #[test]
    #[should_fail]
    fn test_radix_base_too_large() {
        let _ = radix(55, 37);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    mod uint {
        use super::test::Bencher;
        use fmt::radix;
        use rand::{XorShiftRng, Rng};

        #[bench]
        fn format_bin(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:t}", rng.gen::<uint>()); })
        }

        #[bench]
        fn format_oct(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:o}", rng.gen::<uint>()); })
        }

        #[bench]
        fn format_dec(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:u}", rng.gen::<uint>()); })
        }

        #[bench]
        fn format_hex(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:x}", rng.gen::<uint>()); })
        }

        #[bench]
        fn format_base_36(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{}", radix(rng.gen::<uint>(), 36)); })
        }
    }

    mod int {
        use super::test::Bencher;
        use fmt::radix;
        use rand::{XorShiftRng, Rng};

        #[bench]
        fn format_bin(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:t}", rng.gen::<int>()); })
        }

        #[bench]
        fn format_oct(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:o}", rng.gen::<int>()); })
        }

        #[bench]
        fn format_dec(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:d}", rng.gen::<int>()); })
        }

        #[bench]
        fn format_hex(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{:x}", rng.gen::<int>()); })
        }

        #[bench]
        fn format_base_36(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| { format!("{}", radix(rng.gen::<int>(), 36)); })
        }
    }
}
