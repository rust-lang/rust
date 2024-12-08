//! Arbitrary-precision decimal class for fallback algorithms.
//!
//! This is only used if the fast-path (native floats) and
//! the Eisel-Lemire algorithm are unable to unambiguously
//! determine the float.
//!
//! The technique used is "Simple Decimal Conversion", developed
//! by Nigel Tao and Ken Thompson. A detailed description of the
//! algorithm can be found in "ParseNumberF64 by Simple Decimal Conversion",
//! available online: <https://nigeltao.github.io/blog/2020/parse-number-f64-simple.html>.

use crate::num::dec2flt::common::{ByteSlice, is_8digits};

#[derive(Clone)]
pub struct Decimal {
    /// The number of significant digits in the decimal.
    pub num_digits: usize,
    /// The offset of the decimal point in the significant digits.
    pub decimal_point: i32,
    /// If the number of significant digits stored in the decimal is truncated.
    pub truncated: bool,
    /// Buffer of the raw digits, in the range [0, 9].
    pub digits: [u8; Self::MAX_DIGITS],
}

impl Default for Decimal {
    fn default() -> Self {
        Self { num_digits: 0, decimal_point: 0, truncated: false, digits: [0; Self::MAX_DIGITS] }
    }
}

impl Decimal {
    /// The maximum number of digits required to unambiguously round a float.
    ///
    /// For a double-precision IEEE 754 float, this required 767 digits,
    /// so we store the max digits + 1.
    ///
    /// We can exactly represent a float in radix `b` from radix 2 if
    /// `b` is divisible by 2. This function calculates the exact number of
    /// digits required to exactly represent that float.
    ///
    /// According to the "Handbook of Floating Point Arithmetic",
    /// for IEEE754, with emin being the min exponent, p2 being the
    /// precision, and b being the radix, the number of digits follows as:
    ///
    /// `−emin + p2 + ⌊(emin + 1) log(2, b) − log(1 − 2^(−p2), b)⌋`
    ///
    /// For f32, this follows as:
    ///     emin = -126
    ///     p2 = 24
    ///
    /// For f64, this follows as:
    ///     emin = -1022
    ///     p2 = 53
    ///
    /// In Python:
    ///     `-emin + p2 + math.floor((emin+ 1)*math.log(2, b)-math.log(1-2**(-p2), b))`
    pub const MAX_DIGITS: usize = 768;
    /// The max digits that can be exactly represented in a 64-bit integer.
    pub const MAX_DIGITS_WITHOUT_OVERFLOW: usize = 19;
    pub const DECIMAL_POINT_RANGE: i32 = 2047;

    /// Append a digit to the buffer.
    pub fn try_add_digit(&mut self, digit: u8) {
        if self.num_digits < Self::MAX_DIGITS {
            self.digits[self.num_digits] = digit;
        }
        self.num_digits += 1;
    }

    /// Trim trailing zeros from the buffer.
    pub fn trim(&mut self) {
        // All of the following calls to `Decimal::trim` can't panic because:
        //
        //  1. `parse_decimal` sets `num_digits` to a max of `Decimal::MAX_DIGITS`.
        //  2. `right_shift` sets `num_digits` to `write_index`, which is bounded by `num_digits`.
        //  3. `left_shift` `num_digits` to a max of `Decimal::MAX_DIGITS`.
        //
        // Trim is only called in `right_shift` and `left_shift`.
        debug_assert!(self.num_digits <= Self::MAX_DIGITS);
        while self.num_digits != 0 && self.digits[self.num_digits - 1] == 0 {
            self.num_digits -= 1;
        }
    }

    pub fn round(&self) -> u64 {
        if self.num_digits == 0 || self.decimal_point < 0 {
            return 0;
        } else if self.decimal_point > 18 {
            return 0xFFFF_FFFF_FFFF_FFFF_u64;
        }
        let dp = self.decimal_point as usize;
        let mut n = 0_u64;
        for i in 0..dp {
            n *= 10;
            if i < self.num_digits {
                n += self.digits[i] as u64;
            }
        }
        let mut round_up = false;
        if dp < self.num_digits {
            round_up = self.digits[dp] >= 5;
            if self.digits[dp] == 5 && dp + 1 == self.num_digits {
                round_up = self.truncated || ((dp != 0) && (1 & self.digits[dp - 1] != 0))
            }
        }
        if round_up {
            n += 1;
        }
        n
    }

    /// Computes decimal * 2^shift.
    pub fn left_shift(&mut self, shift: usize) {
        if self.num_digits == 0 {
            return;
        }
        let num_new_digits = number_of_digits_decimal_left_shift(self, shift);
        let mut read_index = self.num_digits;
        let mut write_index = self.num_digits + num_new_digits;
        let mut n = 0_u64;
        while read_index != 0 {
            read_index -= 1;
            write_index -= 1;
            n += (self.digits[read_index] as u64) << shift;
            let quotient = n / 10;
            let remainder = n - (10 * quotient);
            if write_index < Self::MAX_DIGITS {
                self.digits[write_index] = remainder as u8;
            } else if remainder > 0 {
                self.truncated = true;
            }
            n = quotient;
        }
        while n > 0 {
            write_index -= 1;
            let quotient = n / 10;
            let remainder = n - (10 * quotient);
            if write_index < Self::MAX_DIGITS {
                self.digits[write_index] = remainder as u8;
            } else if remainder > 0 {
                self.truncated = true;
            }
            n = quotient;
        }
        self.num_digits += num_new_digits;
        if self.num_digits > Self::MAX_DIGITS {
            self.num_digits = Self::MAX_DIGITS;
        }
        self.decimal_point += num_new_digits as i32;
        self.trim();
    }

    /// Computes decimal * 2^-shift.
    pub fn right_shift(&mut self, shift: usize) {
        let mut read_index = 0;
        let mut write_index = 0;
        let mut n = 0_u64;
        while (n >> shift) == 0 {
            if read_index < self.num_digits {
                n = (10 * n) + self.digits[read_index] as u64;
                read_index += 1;
            } else if n == 0 {
                return;
            } else {
                while (n >> shift) == 0 {
                    n *= 10;
                    read_index += 1;
                }
                break;
            }
        }
        self.decimal_point -= read_index as i32 - 1;
        if self.decimal_point < -Self::DECIMAL_POINT_RANGE {
            // `self = Self::Default()`, but without the overhead of clearing `digits`.
            self.num_digits = 0;
            self.decimal_point = 0;
            self.truncated = false;
            return;
        }
        let mask = (1_u64 << shift) - 1;
        while read_index < self.num_digits {
            let new_digit = (n >> shift) as u8;
            n = (10 * (n & mask)) + self.digits[read_index] as u64;
            read_index += 1;
            self.digits[write_index] = new_digit;
            write_index += 1;
        }
        while n > 0 {
            let new_digit = (n >> shift) as u8;
            n = 10 * (n & mask);
            if write_index < Self::MAX_DIGITS {
                self.digits[write_index] = new_digit;
                write_index += 1;
            } else if new_digit > 0 {
                self.truncated = true;
            }
        }
        self.num_digits = write_index;
        self.trim();
    }
}

/// Parse a big integer representation of the float as a decimal.
pub fn parse_decimal(mut s: &[u8]) -> Decimal {
    let mut d = Decimal::default();
    let start = s;

    while let Some((&b'0', s_next)) = s.split_first() {
        s = s_next;
    }

    s = s.parse_digits(|digit| d.try_add_digit(digit));

    if let Some((b'.', s_next)) = s.split_first() {
        s = s_next;
        let first = s;
        // Skip leading zeros.
        if d.num_digits == 0 {
            while let Some((&b'0', s_next)) = s.split_first() {
                s = s_next;
            }
        }
        while s.len() >= 8 && d.num_digits + 8 < Decimal::MAX_DIGITS {
            let v = s.read_u64();
            if !is_8digits(v) {
                break;
            }
            d.digits[d.num_digits..].write_u64(v - 0x3030_3030_3030_3030);
            d.num_digits += 8;
            s = &s[8..];
        }
        s = s.parse_digits(|digit| d.try_add_digit(digit));
        d.decimal_point = s.len() as i32 - first.len() as i32;
    }
    if d.num_digits != 0 {
        // Ignore the trailing zeros if there are any
        let mut n_trailing_zeros = 0;
        for &c in start[..(start.len() - s.len())].iter().rev() {
            if c == b'0' {
                n_trailing_zeros += 1;
            } else if c != b'.' {
                break;
            }
        }
        d.decimal_point += n_trailing_zeros as i32;
        d.num_digits -= n_trailing_zeros;
        d.decimal_point += d.num_digits as i32;
        if d.num_digits > Decimal::MAX_DIGITS {
            d.truncated = true;
            d.num_digits = Decimal::MAX_DIGITS;
        }
    }
    if let Some((&ch, s_next)) = s.split_first() {
        if ch == b'e' || ch == b'E' {
            s = s_next;
            let mut neg_exp = false;
            if let Some((&ch, s_next)) = s.split_first() {
                neg_exp = ch == b'-';
                if ch == b'-' || ch == b'+' {
                    s = s_next;
                }
            }
            let mut exp_num = 0_i32;

            s.parse_digits(|digit| {
                if exp_num < 0x10000 {
                    exp_num = 10 * exp_num + digit as i32;
                }
            });

            d.decimal_point += if neg_exp { -exp_num } else { exp_num };
        }
    }
    for i in d.num_digits..Decimal::MAX_DIGITS_WITHOUT_OVERFLOW {
        d.digits[i] = 0;
    }
    d
}

fn number_of_digits_decimal_left_shift(d: &Decimal, mut shift: usize) -> usize {
    #[rustfmt::skip]
    const TABLE: [u16; 65] = [
        0x0000, 0x0800, 0x0801, 0x0803, 0x1006, 0x1009, 0x100D, 0x1812, 0x1817, 0x181D, 0x2024,
        0x202B, 0x2033, 0x203C, 0x2846, 0x2850, 0x285B, 0x3067, 0x3073, 0x3080, 0x388E, 0x389C,
        0x38AB, 0x38BB, 0x40CC, 0x40DD, 0x40EF, 0x4902, 0x4915, 0x4929, 0x513E, 0x5153, 0x5169,
        0x5180, 0x5998, 0x59B0, 0x59C9, 0x61E3, 0x61FD, 0x6218, 0x6A34, 0x6A50, 0x6A6D, 0x6A8B,
        0x72AA, 0x72C9, 0x72E9, 0x7B0A, 0x7B2B, 0x7B4D, 0x8370, 0x8393, 0x83B7, 0x83DC, 0x8C02,
        0x8C28, 0x8C4F, 0x9477, 0x949F, 0x94C8, 0x9CF2, 0x051C, 0x051C, 0x051C, 0x051C,
    ];
    #[rustfmt::skip]
    const TABLE_POW5: [u8; 0x051C] = [
        5, 2, 5, 1, 2, 5, 6, 2, 5, 3, 1, 2, 5, 1, 5, 6, 2, 5, 7, 8, 1, 2, 5, 3, 9, 0, 6, 2, 5, 1,
        9, 5, 3, 1, 2, 5, 9, 7, 6, 5, 6, 2, 5, 4, 8, 8, 2, 8, 1, 2, 5, 2, 4, 4, 1, 4, 0, 6, 2, 5,
        1, 2, 2, 0, 7, 0, 3, 1, 2, 5, 6, 1, 0, 3, 5, 1, 5, 6, 2, 5, 3, 0, 5, 1, 7, 5, 7, 8, 1, 2,
        5, 1, 5, 2, 5, 8, 7, 8, 9, 0, 6, 2, 5, 7, 6, 2, 9, 3, 9, 4, 5, 3, 1, 2, 5, 3, 8, 1, 4, 6,
        9, 7, 2, 6, 5, 6, 2, 5, 1, 9, 0, 7, 3, 4, 8, 6, 3, 2, 8, 1, 2, 5, 9, 5, 3, 6, 7, 4, 3, 1,
        6, 4, 0, 6, 2, 5, 4, 7, 6, 8, 3, 7, 1, 5, 8, 2, 0, 3, 1, 2, 5, 2, 3, 8, 4, 1, 8, 5, 7, 9,
        1, 0, 1, 5, 6, 2, 5, 1, 1, 9, 2, 0, 9, 2, 8, 9, 5, 5, 0, 7, 8, 1, 2, 5, 5, 9, 6, 0, 4, 6,
        4, 4, 7, 7, 5, 3, 9, 0, 6, 2, 5, 2, 9, 8, 0, 2, 3, 2, 2, 3, 8, 7, 6, 9, 5, 3, 1, 2, 5, 1,
        4, 9, 0, 1, 1, 6, 1, 1, 9, 3, 8, 4, 7, 6, 5, 6, 2, 5, 7, 4, 5, 0, 5, 8, 0, 5, 9, 6, 9, 2,
        3, 8, 2, 8, 1, 2, 5, 3, 7, 2, 5, 2, 9, 0, 2, 9, 8, 4, 6, 1, 9, 1, 4, 0, 6, 2, 5, 1, 8, 6,
        2, 6, 4, 5, 1, 4, 9, 2, 3, 0, 9, 5, 7, 0, 3, 1, 2, 5, 9, 3, 1, 3, 2, 2, 5, 7, 4, 6, 1, 5,
        4, 7, 8, 5, 1, 5, 6, 2, 5, 4, 6, 5, 6, 6, 1, 2, 8, 7, 3, 0, 7, 7, 3, 9, 2, 5, 7, 8, 1, 2,
        5, 2, 3, 2, 8, 3, 0, 6, 4, 3, 6, 5, 3, 8, 6, 9, 6, 2, 8, 9, 0, 6, 2, 5, 1, 1, 6, 4, 1, 5,
        3, 2, 1, 8, 2, 6, 9, 3, 4, 8, 1, 4, 4, 5, 3, 1, 2, 5, 5, 8, 2, 0, 7, 6, 6, 0, 9, 1, 3, 4,
        6, 7, 4, 0, 7, 2, 2, 6, 5, 6, 2, 5, 2, 9, 1, 0, 3, 8, 3, 0, 4, 5, 6, 7, 3, 3, 7, 0, 3, 6,
        1, 3, 2, 8, 1, 2, 5, 1, 4, 5, 5, 1, 9, 1, 5, 2, 2, 8, 3, 6, 6, 8, 5, 1, 8, 0, 6, 6, 4, 0,
        6, 2, 5, 7, 2, 7, 5, 9, 5, 7, 6, 1, 4, 1, 8, 3, 4, 2, 5, 9, 0, 3, 3, 2, 0, 3, 1, 2, 5, 3,
        6, 3, 7, 9, 7, 8, 8, 0, 7, 0, 9, 1, 7, 1, 2, 9, 5, 1, 6, 6, 0, 1, 5, 6, 2, 5, 1, 8, 1, 8,
        9, 8, 9, 4, 0, 3, 5, 4, 5, 8, 5, 6, 4, 7, 5, 8, 3, 0, 0, 7, 8, 1, 2, 5, 9, 0, 9, 4, 9, 4,
        7, 0, 1, 7, 7, 2, 9, 2, 8, 2, 3, 7, 9, 1, 5, 0, 3, 9, 0, 6, 2, 5, 4, 5, 4, 7, 4, 7, 3, 5,
        0, 8, 8, 6, 4, 6, 4, 1, 1, 8, 9, 5, 7, 5, 1, 9, 5, 3, 1, 2, 5, 2, 2, 7, 3, 7, 3, 6, 7, 5,
        4, 4, 3, 2, 3, 2, 0, 5, 9, 4, 7, 8, 7, 5, 9, 7, 6, 5, 6, 2, 5, 1, 1, 3, 6, 8, 6, 8, 3, 7,
        7, 2, 1, 6, 1, 6, 0, 2, 9, 7, 3, 9, 3, 7, 9, 8, 8, 2, 8, 1, 2, 5, 5, 6, 8, 4, 3, 4, 1, 8,
        8, 6, 0, 8, 0, 8, 0, 1, 4, 8, 6, 9, 6, 8, 9, 9, 4, 1, 4, 0, 6, 2, 5, 2, 8, 4, 2, 1, 7, 0,
        9, 4, 3, 0, 4, 0, 4, 0, 0, 7, 4, 3, 4, 8, 4, 4, 9, 7, 0, 7, 0, 3, 1, 2, 5, 1, 4, 2, 1, 0,
        8, 5, 4, 7, 1, 5, 2, 0, 2, 0, 0, 3, 7, 1, 7, 4, 2, 2, 4, 8, 5, 3, 5, 1, 5, 6, 2, 5, 7, 1,
        0, 5, 4, 2, 7, 3, 5, 7, 6, 0, 1, 0, 0, 1, 8, 5, 8, 7, 1, 1, 2, 4, 2, 6, 7, 5, 7, 8, 1, 2,
        5, 3, 5, 5, 2, 7, 1, 3, 6, 7, 8, 8, 0, 0, 5, 0, 0, 9, 2, 9, 3, 5, 5, 6, 2, 1, 3, 3, 7, 8,
        9, 0, 6, 2, 5, 1, 7, 7, 6, 3, 5, 6, 8, 3, 9, 4, 0, 0, 2, 5, 0, 4, 6, 4, 6, 7, 7, 8, 1, 0,
        6, 6, 8, 9, 4, 5, 3, 1, 2, 5, 8, 8, 8, 1, 7, 8, 4, 1, 9, 7, 0, 0, 1, 2, 5, 2, 3, 2, 3, 3,
        8, 9, 0, 5, 3, 3, 4, 4, 7, 2, 6, 5, 6, 2, 5, 4, 4, 4, 0, 8, 9, 2, 0, 9, 8, 5, 0, 0, 6, 2,
        6, 1, 6, 1, 6, 9, 4, 5, 2, 6, 6, 7, 2, 3, 6, 3, 2, 8, 1, 2, 5, 2, 2, 2, 0, 4, 4, 6, 0, 4,
        9, 2, 5, 0, 3, 1, 3, 0, 8, 0, 8, 4, 7, 2, 6, 3, 3, 3, 6, 1, 8, 1, 6, 4, 0, 6, 2, 5, 1, 1,
        1, 0, 2, 2, 3, 0, 2, 4, 6, 2, 5, 1, 5, 6, 5, 4, 0, 4, 2, 3, 6, 3, 1, 6, 6, 8, 0, 9, 0, 8,
        2, 0, 3, 1, 2, 5, 5, 5, 5, 1, 1, 1, 5, 1, 2, 3, 1, 2, 5, 7, 8, 2, 7, 0, 2, 1, 1, 8, 1, 5,
        8, 3, 4, 0, 4, 5, 4, 1, 0, 1, 5, 6, 2, 5, 2, 7, 7, 5, 5, 5, 7, 5, 6, 1, 5, 6, 2, 8, 9, 1,
        3, 5, 1, 0, 5, 9, 0, 7, 9, 1, 7, 0, 2, 2, 7, 0, 5, 0, 7, 8, 1, 2, 5, 1, 3, 8, 7, 7, 7, 8,
        7, 8, 0, 7, 8, 1, 4, 4, 5, 6, 7, 5, 5, 2, 9, 5, 3, 9, 5, 8, 5, 1, 1, 3, 5, 2, 5, 3, 9, 0,
        6, 2, 5, 6, 9, 3, 8, 8, 9, 3, 9, 0, 3, 9, 0, 7, 2, 2, 8, 3, 7, 7, 6, 4, 7, 6, 9, 7, 9, 2,
        5, 5, 6, 7, 6, 2, 6, 9, 5, 3, 1, 2, 5, 3, 4, 6, 9, 4, 4, 6, 9, 5, 1, 9, 5, 3, 6, 1, 4, 1,
        8, 8, 8, 2, 3, 8, 4, 8, 9, 6, 2, 7, 8, 3, 8, 1, 3, 4, 7, 6, 5, 6, 2, 5, 1, 7, 3, 4, 7, 2,
        3, 4, 7, 5, 9, 7, 6, 8, 0, 7, 0, 9, 4, 4, 1, 1, 9, 2, 4, 4, 8, 1, 3, 9, 1, 9, 0, 6, 7, 3,
        8, 2, 8, 1, 2, 5, 8, 6, 7, 3, 6, 1, 7, 3, 7, 9, 8, 8, 4, 0, 3, 5, 4, 7, 2, 0, 5, 9, 6, 2,
        2, 4, 0, 6, 9, 5, 9, 5, 3, 3, 6, 9, 1, 4, 0, 6, 2, 5,
    ];

    shift &= 63;
    let x_a = TABLE[shift];
    let x_b = TABLE[shift + 1];
    let num_new_digits = (x_a >> 11) as _;
    let pow5_a = (0x7FF & x_a) as usize;
    let pow5_b = (0x7FF & x_b) as usize;
    let pow5 = &TABLE_POW5[pow5_a..];
    for (i, &p5) in pow5.iter().enumerate().take(pow5_b - pow5_a) {
        if i >= d.num_digits {
            return num_new_digits - 1;
        } else if d.digits[i] == p5 {
            continue;
        } else if d.digits[i] < p5 {
            return num_new_digits - 1;
        } else {
            return num_new_digits;
        }
    }
    num_new_digits
}
