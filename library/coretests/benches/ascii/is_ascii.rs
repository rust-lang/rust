use test::{Bencher, black_box};

use super::{LONG, MEDIUM, SHORT};

macro_rules! benches {
    ($( fn $name: ident($arg: ident: &[u8]) $body: block )+) => {
        benches!(mod short SHORT[..] $($name $arg $body)+);
        benches!(mod medium MEDIUM[..] $($name $arg $body)+);
        benches!(mod medium_15 MEDIUM[..=15] $($name $arg $body)+);
        benches!(mod long LONG[..] $($name $arg $body)+);
        // Ensure we benchmark cases where the functions are called with strings
        // that are not perfectly aligned or have a length which is not a
        // multiple of size_of::<usize>() (or both)
        benches!(mod unaligned_head_medium MEDIUM[1..] $($name $arg $body)+);
        benches!(mod unaligned_tail_medium MEDIUM[..(MEDIUM.len() - 1)] $($name $arg $body)+);
        benches!(mod unaligned_both_medium MEDIUM[1..(MEDIUM.len() - 1)] $($name $arg $body)+);
        benches!(mod unaligned_head_long LONG[1..] $($name $arg $body)+);
        benches!(mod unaligned_tail_long LONG[..(LONG.len() - 1)] $($name $arg $body)+);
        benches!(mod unaligned_both_long LONG[1..(LONG.len() - 1)] $($name $arg $body)+);
    };

    (mod $mod_name: ident $input: ident [$range: expr] $($name: ident $arg: ident $body: block)+) => {
        mod $mod_name {
            use super::*;
            $(
                #[bench]
                fn $name(bencher: &mut Bencher) {
                    bencher.bytes = $input[$range].len() as u64;
                    let mut vec = $input.as_bytes().to_vec();
                    bencher.iter(|| {
                        let $arg: &[u8] = &black_box(&mut vec)[$range];
                        black_box($body)
                    })
                }
            )+
        }
    };
}

benches! {
    fn is_ascii_swar_1(bytes: &[u8]) {
        core::slice::is_ascii_swar::<1>(bytes)
    }

    fn is_ascii_swar_2(bytes: &[u8]) {
        core::slice::is_ascii_swar::<2>(bytes)
    }

    fn is_ascii_swar_4(bytes: &[u8]) {
        core::slice::is_ascii_swar::<4>(bytes)
    }

    fn is_ascii_simd_08(bytes: &[u8]) {
        core::slice::is_ascii_simd::<8>(bytes)
    }

    fn is_ascii_simd_16(bytes: &[u8]) {
        core::slice::is_ascii_simd::<16>(bytes)
    }

    fn is_ascii_simd_32(bytes: &[u8]) {
        core::slice::is_ascii_simd::<32>(bytes)
    }
}
