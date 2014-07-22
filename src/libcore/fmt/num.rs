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

use collections::Collection;
use fmt;
use iter::DoubleEndedIterator;
use num::{Int, cast, zero};
use slice::{ImmutableVector, MutableVector};

#[cfg(stage0)]
use iter::Iterator;         // NOTE(stage0): Remove after snapshot.
#[cfg(stage0)]
use option::{Some, None};   // NOTE(stage0): Remove after snapshot.

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
#[deriving(Clone, PartialEq)]
struct Binary;

/// An octal (base 8) radix
#[deriving(Clone, PartialEq)]
struct Octal;

/// A decimal (base 10) radix
#[deriving(Clone, PartialEq)]
struct Decimal;

/// A hexadecimal (base 16) radix, formatted with lower-case characters
#[deriving(Clone, PartialEq)]
struct LowerHex;

/// A hexadecimal (base 16) radix, formatted with upper-case characters
#[deriving(Clone, PartialEq)]
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
#[deriving(Clone, PartialEq)]
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
/// assert_eq!(format!("{}", radix(55i, 36)), "1j".to_string());
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
