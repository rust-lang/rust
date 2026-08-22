mod dec2flt;
mod flt2dec;
mod int_bits;
mod int_log;
mod int_pow;
mod int_sqrt;

use std::str::FromStr;

use test::{Bencher, black_box};

const ASCII_NUMBERS: [&str; 19] = [
    "0",
    "1",
    "2",
    "43",
    "765",
    "76567",
    "987245987",
    "-4aa32",
    "1786235",
    "8723095",
    "f##5s",
    "83638730",
    "-2345",
    "562aa43",
    "-1",
    "-0",
    "abc",
    "xyz",
    "c0ffee",
];

/// Long decimal strings (16-20 digits) that trigger the SWAR fast path
/// multiple times for 64-bit integer parsing.
const LONG_ASCII_NUMBERS: [&str; 10] = [
    "1234567890123456",     // 16 digits, exactly 2 SWAR chunks
    "12345678901234567",    // 17 digits
    "123456789012345678",   // 18 digits
    "1234567890123456789",  // 19 digits
    "18446744073709551615", // 20 digits, u64::MAX
    "9223372036854775807",  // 19 digits, i64::MAX
    "9999999999999999",     // 16 digits
    "10000000000000000",    // 17 digits
    "-9223372036854775808", // 19 digits + sign, i64::MIN
    "0000123456789012",     // 16 digits with leading zeros
];

macro_rules! from_str_bench {
    ($mac:ident, $t:ty) => {
        #[bench]
        fn $mac(b: &mut Bencher) {
            b.iter(|| {
                ASCII_NUMBERS
                    .iter()
                    .cycle()
                    .take(5_000)
                    .filter_map(|s| <$t>::from_str(black_box(s)).ok())
                    .max()
            })
        }
    };
}

macro_rules! from_str_radix_bench {
    ($mac:ident, $t:ty, $radix:expr) => {
        #[bench]
        fn $mac(b: &mut Bencher) {
            b.iter(|| {
                ASCII_NUMBERS
                    .iter()
                    .cycle()
                    .take(5_000)
                    .filter_map(|s| <$t>::from_str_radix(black_box(s), $radix).ok())
                    .max()
            })
        }
    };
}

macro_rules! from_str_radix_long_bench {
    ($mac:ident, $t:ty, $radix:expr) => {
        #[bench]
        fn $mac(b: &mut Bencher) {
            b.iter(|| {
                LONG_ASCII_NUMBERS
                    .iter()
                    .cycle()
                    .take(5_000)
                    .filter_map(|s| <$t>::from_str_radix(black_box(s), $radix).ok())
                    .max()
            })
        }
    };
}

from_str_bench!(bench_u8_from_str, u8);
from_str_radix_bench!(bench_u8_from_str_radix_2, u8, 2);
from_str_radix_bench!(bench_u8_from_str_radix_10, u8, 10);
from_str_radix_bench!(bench_u8_from_str_radix_16, u8, 16);
from_str_radix_bench!(bench_u8_from_str_radix_36, u8, 36);

from_str_bench!(bench_u16_from_str, u16);
from_str_radix_bench!(bench_u16_from_str_radix_2, u16, 2);
from_str_radix_bench!(bench_u16_from_str_radix_10, u16, 10);
from_str_radix_bench!(bench_u16_from_str_radix_16, u16, 16);
from_str_radix_bench!(bench_u16_from_str_radix_36, u16, 36);

from_str_bench!(bench_u32_from_str, u32);
from_str_radix_bench!(bench_u32_from_str_radix_2, u32, 2);
from_str_radix_bench!(bench_u32_from_str_radix_10, u32, 10);
from_str_radix_bench!(bench_u32_from_str_radix_16, u32, 16);
from_str_radix_bench!(bench_u32_from_str_radix_36, u32, 36);

from_str_bench!(bench_u64_from_str, u64);
from_str_radix_bench!(bench_u64_from_str_radix_2, u64, 2);
from_str_radix_bench!(bench_u64_from_str_radix_10, u64, 10);
from_str_radix_bench!(bench_u64_from_str_radix_16, u64, 16);
from_str_radix_bench!(bench_u64_from_str_radix_36, u64, 36);

from_str_bench!(bench_i8_from_str, i8);
from_str_radix_bench!(bench_i8_from_str_radix_2, i8, 2);
from_str_radix_bench!(bench_i8_from_str_radix_10, i8, 10);
from_str_radix_bench!(bench_i8_from_str_radix_16, i8, 16);
from_str_radix_bench!(bench_i8_from_str_radix_36, i8, 36);

from_str_bench!(bench_i16_from_str, i16);
from_str_radix_bench!(bench_i16_from_str_radix_2, i16, 2);
from_str_radix_bench!(bench_i16_from_str_radix_10, i16, 10);
from_str_radix_bench!(bench_i16_from_str_radix_16, i16, 16);
from_str_radix_bench!(bench_i16_from_str_radix_36, i16, 36);

from_str_bench!(bench_i32_from_str, i32);
from_str_radix_bench!(bench_i32_from_str_radix_2, i32, 2);
from_str_radix_bench!(bench_i32_from_str_radix_10, i32, 10);
from_str_radix_bench!(bench_i32_from_str_radix_16, i32, 16);
from_str_radix_bench!(bench_i32_from_str_radix_36, i32, 36);

from_str_bench!(bench_i64_from_str, i64);
from_str_radix_bench!(bench_i64_from_str_radix_2, i64, 2);
from_str_radix_bench!(bench_i64_from_str_radix_10, i64, 10);
from_str_radix_bench!(bench_i64_from_str_radix_16, i64, 16);
from_str_radix_bench!(bench_i64_from_str_radix_36, i64, 36);

// Long-string benchmarks: 16-20 digit decimal numbers that exercise
// the SWAR fast path (2+ iterations of 8-digit-at-a-time parsing).
from_str_radix_long_bench!(bench_u64_from_str_radix_10_long, u64, 10);
from_str_radix_long_bench!(bench_i64_from_str_radix_10_long, i64, 10);
