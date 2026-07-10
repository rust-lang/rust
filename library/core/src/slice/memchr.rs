// Original implementation taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

use crate::ffi::{c_int, c_void};
use crate::intrinsics::const_eval_select;

const LO_USIZE: usize = usize::repeat_u8(0x01);
const HI_USIZE: usize = usize::repeat_u8(0x80);

/// Returns `true` if `x` contains any zero byte.
///
/// From *Matters Computational*, J. Arndt:
///
/// "The idea is to subtract one from each of the bytes and then look for
/// bytes where the borrow propagated all the way to the most significant
/// bit."
#[inline]
const fn contains_zero_byte(x: usize) -> bool {
    x.wrapping_sub(LO_USIZE) & !x & HI_USIZE != 0
}

/// Returns the first index matching the byte `x` in `text`.
#[inline]
#[must_use]
#[rustc_allow_const_fn_unstable(const_eval_select)] // both impls have the exact same behaviour
pub const fn memchr(x: u8, text: &[u8]) -> Option<usize> {
    const_eval_select!(
        @capture { x: u8 = x, text: &[u8] = text } -> Option<usize>:
        if const {
            let mut i = 0;

            // FIXME(const-hack): Replace with `text.iter().pos(|c| *c == x)`.
            while i < text.len() {
                if text[i] == x {
                    return Some(i);
                }

                i += 1;
            }

            None
        } else {
            unsafe extern "C" {
                // Provided in either libc or compiler-builtins.
                fn memchr(
                    s: *const c_void,
                    c: c_int,
                    n: usize,
                ) -> *mut c_void;
            }

            // SAFETY:
            // The pointer and length come from a slice reference and thus
            // describe a valid memory region. Since the reference is a `&[u8]`,
            // every byte contained therein is interpretable as an initialized
            // byte.
            let res = unsafe { memchr(text.as_ptr().cast(), x as c_int, text.len()) };
            if res.is_null() {
                None
            } else {
                // SAFETY: `res` is non-null, and thus is guaranteed to lie
                // within the memory region passed to `memchr`.
                let index = unsafe { res.offset_from_unsigned(ptr) };
                Some(index)
            }
        }
    )
}

/// Returns the last index matching the byte `x` in `text`.
#[must_use]
pub fn memrchr(x: u8, text: &[u8]) -> Option<usize> {
    // Scan for a single byte value by reading two `usize` words at a time.
    //
    // Split `text` in three parts:
    // - unaligned tail, after the last word aligned address in text,
    // - body, scanned by 2 words at a time,
    // - the first remaining bytes, < 2 word size.
    let len = text.len();
    let ptr = text.as_ptr();
    type Chunk = usize;

    let (min_aligned_offset, max_aligned_offset) = {
        // We call this just to obtain the length of the prefix and suffix.
        // In the middle we always process two chunks at once.
        // SAFETY: transmuting `[u8]` to `[usize]` is safe except for size differences
        // which are handled by `align_to`.
        let (prefix, _, suffix) = unsafe { text.align_to::<(Chunk, Chunk)>() };
        (prefix.len(), len - suffix.len())
    };

    let mut offset = max_aligned_offset;
    if let Some(index) = text[offset..].iter().rposition(|elt| *elt == x) {
        return Some(offset + index);
    }

    // Search the body of the text, make sure we don't cross min_aligned_offset.
    // offset is always aligned, so just testing `>` is sufficient and avoids possible
    // overflow.
    let repeated_x = usize::repeat_u8(x);
    let chunk_bytes = size_of::<Chunk>();

    while offset > min_aligned_offset {
        // SAFETY: offset starts at len - suffix.len(), as long as it is greater than
        // min_aligned_offset (prefix.len()) the remaining distance is at least 2 * chunk_bytes.
        unsafe {
            let u = *(ptr.add(offset - 2 * chunk_bytes) as *const Chunk);
            let v = *(ptr.add(offset - chunk_bytes) as *const Chunk);

            // Break if there is a matching byte.
            let zu = contains_zero_byte(u ^ repeated_x);
            let zv = contains_zero_byte(v ^ repeated_x);
            if zu || zv {
                break;
            }
        }
        offset -= 2 * chunk_bytes;
    }

    // Find the byte before the point the body loop stopped.
    text[..offset].iter().rposition(|elt| *elt == x)
}
