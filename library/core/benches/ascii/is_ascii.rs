use test::{Bencher, black_box};

use super::{LONG, MEDIUM, SHORT};

macro_rules! benches {
    ($( fn $name: ident($arg: ident: &[u8]) $body: block )+) => {
        benches!(mod short SHORT[..] $($name $arg $body)+);
        benches!(mod medium MEDIUM[..] $($name $arg $body)+);
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
    fn case00_libcore(bytes: &[u8]) {
        bytes.is_ascii()
    }

    fn case01_iter_all(bytes: &[u8]) {
        bytes.iter().all(|b| b.is_ascii())
    }

    fn case02_align_to(bytes: &[u8]) {
        is_ascii_align_to(bytes)
    }

    fn case03_align_to_unrolled(bytes: &[u8]) {
        is_ascii_align_to_unrolled(bytes)
    }

    fn case04_while_loop(bytes: &[u8]) {
        // Process chunks of 32 bytes at a time in the fast path to enable
        // auto-vectorization and use of `pmovmskb`. Two 128-bit vector registers
        // can be OR'd together and then the resulting vector can be tested for
        // non-ASCII bytes.
        const CHUNK_SIZE: usize = 32;

        let mut i = 0;

        while i + CHUNK_SIZE <= bytes.len() {
            let chunk_end = i + CHUNK_SIZE;

            // Get LLVM to produce a `pmovmskb` instruction on x86-64 which
            // creates a mask from the most significant bit of each byte.
            // ASCII bytes are less than 128 (0x80), so their most significant
            // bit is unset.
            let mut count = 0;
            while i < chunk_end {
                count += bytes[i].is_ascii() as u8;
                i += 1;
            }

            // All bytes should be <= 127 so count is equal to chunk size.
            if count != CHUNK_SIZE as u8 {
                return false;
            }
        }

        // Process the remaining `bytes.len() % N` bytes.
        let mut is_ascii = true;
        while i < bytes.len() {
            is_ascii &= bytes[i].is_ascii();
            i += 1;
        }

        is_ascii
    }
}

// These are separate since it's easier to debug errors if they don't go through
// macro expansion first.
fn is_ascii_align_to(bytes: &[u8]) -> bool {
    if bytes.len() < core::mem::size_of::<usize>() {
        return bytes.iter().all(|b| b.is_ascii());
    }
    // SAFETY: transmuting a sequence of `u8` to `usize` is always fine
    let (head, body, tail) = unsafe { bytes.align_to::<usize>() };
    head.iter().all(|b| b.is_ascii())
        && body.iter().all(|w| !contains_nonascii(*w))
        && tail.iter().all(|b| b.is_ascii())
}

fn is_ascii_align_to_unrolled(bytes: &[u8]) -> bool {
    if bytes.len() < core::mem::size_of::<usize>() {
        return bytes.iter().all(|b| b.is_ascii());
    }
    // SAFETY: transmuting a sequence of `u8` to `[usize; 2]` is always fine
    let (head, body, tail) = unsafe { bytes.align_to::<[usize; 2]>() };
    head.iter().all(|b| b.is_ascii())
        && body.iter().all(|w| !contains_nonascii(w[0] | w[1]))
        && tail.iter().all(|b| b.is_ascii())
}

#[inline]
fn contains_nonascii(v: usize) -> bool {
    const NONASCII_MASK: usize = usize::from_ne_bytes([0x80; core::mem::size_of::<usize>()]);
    (NONASCII_MASK & v) != 0
}
