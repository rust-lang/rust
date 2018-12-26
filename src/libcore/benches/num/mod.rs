mod flt2dec;
mod dec2flt;

use test::Bencher;
use std::str::FromStr;

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

macro_rules! from_str_bench {
    ($mac:ident, $t:ty) => (
        #[bench]
        fn $mac(b: &mut Bencher) {
            b.iter(|| {
                ASCII_NUMBERS
                    .iter()
                    .cycle()
                    .take(5_000)
                    .filter_map(|s| <($t)>::from_str(s).ok())
                    .max()
            })
        }
    )
}

macro_rules! from_str_radix_bench {
    ($mac:ident, $t:ty, $radix:expr) => (
        #[bench]
        fn $mac(b: &mut Bencher) {
            b.iter(|| {
                ASCII_NUMBERS
                    .iter()
                    .cycle()
                    .take(5_000)
                    .filter_map(|s| <($t)>::from_str_radix(s, $radix).ok())
                    .max()
            })
        }
    )
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
