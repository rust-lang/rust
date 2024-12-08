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
        benches!(mod unaligned_head MEDIUM[1..] $($name $arg $body)+);
        benches!(mod unaligned_tail MEDIUM[..(MEDIUM.len() - 1)] $($name $arg $body)+);
        benches!(mod unaligned_both MEDIUM[1..(MEDIUM.len() - 1)] $($name $arg $body)+);
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
