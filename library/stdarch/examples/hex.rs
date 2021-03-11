//! An example showing runtime dispatch to an architecture-optimized
//! implementation.
//!
//! This program implements hex encoding a slice into a predetermined
//! destination using various different instruction sets. This selects at
//! runtime the most optimized implementation and uses that rather than being
//! required to be compiled differently.
//!
//! You can test out this program via:
//!
//!     echo test | cargo +nightly run --release hex
//!
//! and you should see `746573740a` get printed out.

#![feature(stdsimd, wasm_target_feature)]
#![cfg_attr(test, feature(test))]
#![cfg_attr(target_arch = "wasm32", feature(wasm_simd))]
#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::shadow_reuse,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::missing_docs_in_private_items
)]

use std::{
    io::{self, Read},
    str,
};

#[cfg(target_arch = "x86")]
use {core_arch::arch::x86::*, std_detect::is_x86_feature_detected};
#[cfg(target_arch = "x86_64")]
use {core_arch::arch::x86_64::*, std_detect::is_x86_feature_detected};

fn main() {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).unwrap();
    let mut dst = vec![0; 2 * input.len()];
    let s = hex_encode(&input, &mut dst).unwrap();
    println!("{}", s);
}

fn hex_encode<'a>(src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, usize> {
    let len = src.len().checked_mul(2).unwrap();
    if dst.len() < len {
        return Err(len);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { hex_encode_avx2(src, dst) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { hex_encode_sse41(src, dst) };
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if true {
            return unsafe { hex_encode_simd128(src, dst) };
        }
    }

    hex_encode_fallback(src, dst)
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hex_encode_avx2<'a>(mut src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, usize> {
    let ascii_zero = _mm256_set1_epi8(b'0' as i8);
    let nines = _mm256_set1_epi8(9);
    let ascii_a = _mm256_set1_epi8((b'a' - 9 - 1) as i8);
    let and4bits = _mm256_set1_epi8(0xf);

    let mut i = 0_isize;
    while src.len() >= 32 {
        let invec = _mm256_loadu_si256(src.as_ptr() as *const _);

        let masked1 = _mm256_and_si256(invec, and4bits);
        let masked2 = _mm256_and_si256(_mm256_srli_epi64(invec, 4), and4bits);

        // return 0xff corresponding to the elements > 9, or 0x00 otherwise
        let cmpmask1 = _mm256_cmpgt_epi8(masked1, nines);
        let cmpmask2 = _mm256_cmpgt_epi8(masked2, nines);

        // add '0' or the offset depending on the masks
        let masked1 = _mm256_add_epi8(masked1, _mm256_blendv_epi8(ascii_zero, ascii_a, cmpmask1));
        let masked2 = _mm256_add_epi8(masked2, _mm256_blendv_epi8(ascii_zero, ascii_a, cmpmask2));

        // interleave masked1 and masked2 bytes
        let res1 = _mm256_unpacklo_epi8(masked2, masked1);
        let res2 = _mm256_unpackhi_epi8(masked2, masked1);

        // Store everything into the right destination now
        let base = dst.as_mut_ptr().offset(i * 2);
        let base1 = base.offset(0) as *mut _;
        let base2 = base.offset(16) as *mut _;
        let base3 = base.offset(32) as *mut _;
        let base4 = base.offset(48) as *mut _;
        _mm256_storeu2_m128i(base3, base1, res1);
        _mm256_storeu2_m128i(base4, base2, res2);
        src = &src[32..];
        i += 32;
    }

    let i = i as usize;
    let _ = hex_encode_sse41(src, &mut dst[i * 2..]);

    Ok(str::from_utf8_unchecked(&dst[..src.len() * 2 + i * 2]))
}

// copied from https://github.com/Matherunner/bin2hex-sse/blob/master/base16_sse4.cpp
#[target_feature(enable = "sse4.1")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hex_encode_sse41<'a>(mut src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, usize> {
    let ascii_zero = _mm_set1_epi8(b'0' as i8);
    let nines = _mm_set1_epi8(9);
    let ascii_a = _mm_set1_epi8((b'a' - 9 - 1) as i8);
    let and4bits = _mm_set1_epi8(0xf);

    let mut i = 0_isize;
    while src.len() >= 16 {
        let invec = _mm_loadu_si128(src.as_ptr() as *const _);

        let masked1 = _mm_and_si128(invec, and4bits);
        let masked2 = _mm_and_si128(_mm_srli_epi64(invec, 4), and4bits);

        // return 0xff corresponding to the elements > 9, or 0x00 otherwise
        let cmpmask1 = _mm_cmpgt_epi8(masked1, nines);
        let cmpmask2 = _mm_cmpgt_epi8(masked2, nines);

        // add '0' or the offset depending on the masks
        let masked1 = _mm_add_epi8(masked1, _mm_blendv_epi8(ascii_zero, ascii_a, cmpmask1));
        let masked2 = _mm_add_epi8(masked2, _mm_blendv_epi8(ascii_zero, ascii_a, cmpmask2));

        // interleave masked1 and masked2 bytes
        let res1 = _mm_unpacklo_epi8(masked2, masked1);
        let res2 = _mm_unpackhi_epi8(masked2, masked1);

        _mm_storeu_si128(dst.as_mut_ptr().offset(i * 2) as *mut _, res1);
        _mm_storeu_si128(dst.as_mut_ptr().offset(i * 2 + 16) as *mut _, res2);
        src = &src[16..];
        i += 16;
    }

    let i = i as usize;
    let _ = hex_encode_fallback(src, &mut dst[i * 2..]);

    Ok(str::from_utf8_unchecked(&dst[..src.len() * 2 + i * 2]))
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn hex_encode_simd128<'a>(mut src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, usize> {
    use core_arch::arch::wasm32::*;

    let ascii_zero = i8x16_splat(b'0' as i8);
    let nines = i8x16_splat(9);
    let ascii_a = i8x16_splat((b'a' - 9 - 1) as i8);
    let and4bits = i8x16_splat(0xf);

    let mut i = 0_isize;
    while src.len() >= 16 {
        let invec = v128_load(src.as_ptr() as *const _);

        let masked1 = v128_and(invec, and4bits);
        let masked2 = v128_and(i8x16_shr_u(invec, 4), and4bits);

        // return 0xff corresponding to the elements > 9, or 0x00 otherwise
        let cmpmask1 = i8x16_gt_u(masked1, nines);
        let cmpmask2 = i8x16_gt_u(masked2, nines);

        // add '0' or the offset depending on the masks
        let masked1 = i8x16_add(masked1, v128_bitselect(ascii_a, ascii_zero, cmpmask1));
        let masked2 = i8x16_add(masked2, v128_bitselect(ascii_a, ascii_zero, cmpmask2));

        // Next we need to shuffle around masked{1,2} to get back to the
        // original source text order. The first element (res1) we'll store uses
        // all the low bytes from the 2 masks and the second element (res2) uses
        // all the upper bytes.
        let res1 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
            masked2, masked1,
        );
        let res2 = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
            masked2, masked1,
        );

        v128_store(dst.as_mut_ptr().offset(i * 2) as *mut _, res1);
        v128_store(dst.as_mut_ptr().offset(i * 2 + 16) as *mut _, res2);
        src = &src[16..];
        i += 16;
    }

    let i = i as usize;
    let _ = hex_encode_fallback(src, &mut dst[i * 2..]);

    Ok(str::from_utf8_unchecked(&dst[..src.len() * 2 + i * 2]))
}

fn hex_encode_fallback<'a>(src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, usize> {
    fn hex(byte: u8) -> u8 {
        static TABLE: &[u8] = b"0123456789abcdef";
        TABLE[byte as usize]
    }

    for (byte, slots) in src.iter().zip(dst.chunks_mut(2)) {
        slots[0] = hex((*byte >> 4) & 0xf);
        slots[1] = hex(*byte & 0xf);
    }

    unsafe { Ok(str::from_utf8_unchecked(&dst[..src.len() * 2])) }
}

// Run these with `cargo +nightly test --example hex -p stdarch`
#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    fn test(input: &[u8], output: &str) {
        let tmp = || vec![0; input.len() * 2];

        assert_eq!(hex_encode_fallback(input, &mut tmp()).unwrap(), output);
        assert_eq!(hex_encode(input, &mut tmp()).unwrap(), output);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if self::is_x86_feature_detected!("avx2") {
                assert_eq!(hex_encode_avx2(input, &mut tmp()).unwrap(), output);
            }
            if self::is_x86_feature_detected!("sse4.1") {
                assert_eq!(hex_encode_sse41(input, &mut tmp()).unwrap(), output);
            }
        }
    }

    #[test]
    fn empty() {
        test(b"", "");
    }

    #[test]
    fn big() {
        test(
            &[0; 1024],
            &iter::repeat('0').take(2048).collect::<String>(),
        );
    }

    #[test]
    fn odd() {
        test(
            &[0; 313],
            &iter::repeat('0').take(313 * 2).collect::<String>(),
        );
    }

    #[test]
    fn avx_works() {
        let mut input = [0; 33];
        input[4] = 3;
        input[16] = 3;
        input[17] = 0x30;
        input[21] = 1;
        input[31] = 0x24;
        test(
            &input,
            "\
             0000000003000000\
             0000000000000000\
             0330000000010000\
             0000000000000024\
             00\
             ",
        );
    }

    quickcheck::quickcheck! {
        fn encode_equals_fallback(input: Vec<u8>) -> bool {
            let mut space1 = vec![0; input.len() * 2];
            let mut space2 = vec![0; input.len() * 2];
            let a = hex_encode(&input, &mut space1).unwrap();
            let b = hex_encode_fallback(&input, &mut space2).unwrap();
            a == b
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fn avx_equals_fallback(input: Vec<u8>) -> bool {
            if !self::is_x86_feature_detected!("avx2") {
                return true
            }
            let mut space1 = vec![0; input.len() * 2];
            let mut space2 = vec![0; input.len() * 2];
            let a = unsafe { hex_encode_avx2(&input, &mut space1).unwrap() };
            let b = hex_encode_fallback(&input, &mut space2).unwrap();
            a == b
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fn sse41_equals_fallback(input: Vec<u8>) -> bool {
            if !self::is_x86_feature_detected!("avx2") {
                return true
            }
            let mut space1 = vec![0; input.len() * 2];
            let mut space2 = vec![0; input.len() * 2];
            let a = unsafe { hex_encode_sse41(&input, &mut space1).unwrap() };
            let b = hex_encode_fallback(&input, &mut space2).unwrap();
            a == b
        }
    }
}

// Run these with `cargo +nightly bench --example hex -p stdarch`
#[cfg(test)]
mod benches {
    extern crate rand;
    extern crate test;

    use self::rand::Rng;

    use super::*;

    const SMALL_LEN: usize = 117;
    const LARGE_LEN: usize = 1 * 1024 * 1024;

    fn doit(
        b: &mut test::Bencher,
        len: usize,
        f: for<'a> unsafe fn(&[u8], &'a mut [u8]) -> Result<&'a str, usize>,
    ) {
        let mut rng = rand::thread_rng();
        let input = std::iter::repeat(())
            .map(|()| rng.gen::<u8>())
            .take(len)
            .collect::<Vec<_>>();
        let mut dst = vec![0; input.len() * 2];
        b.bytes = len as u64;
        b.iter(|| unsafe {
            f(&input, &mut dst).unwrap();
            dst[0]
        });
    }

    #[bench]
    fn small_default(b: &mut test::Bencher) {
        doit(b, SMALL_LEN, hex_encode);
    }

    #[bench]
    fn small_fallback(b: &mut test::Bencher) {
        doit(b, SMALL_LEN, hex_encode_fallback);
    }

    #[bench]
    fn large_default(b: &mut test::Bencher) {
        doit(b, LARGE_LEN, hex_encode);
    }

    #[bench]
    fn large_fallback(b: &mut test::Bencher) {
        doit(b, LARGE_LEN, hex_encode_fallback);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod x86 {
        use super::*;

        #[bench]
        fn small_avx2(b: &mut test::Bencher) {
            if self::is_x86_feature_detected!("avx2") {
                doit(b, SMALL_LEN, hex_encode_avx2);
            }
        }

        #[bench]
        fn small_sse41(b: &mut test::Bencher) {
            if self::is_x86_feature_detected!("sse4.1") {
                doit(b, SMALL_LEN, hex_encode_sse41);
            }
        }

        #[bench]
        fn large_avx2(b: &mut test::Bencher) {
            if self::is_x86_feature_detected!("avx2") {
                doit(b, LARGE_LEN, hex_encode_avx2);
            }
        }

        #[bench]
        fn large_sse41(b: &mut test::Bencher) {
            if self::is_x86_feature_detected!("sse4.1") {
                doit(b, LARGE_LEN, hex_encode_sse41);
            }
        }
    }
}
