// Original implementation taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

use crate::cmp;
use crate::mem;

const LO_U64: u64 = 0x0101010101010101;
const HI_U64: u64 = 0x8080808080808080;

// Use truncation.
const LO_USIZE: usize = LO_U64 as usize;
const HI_USIZE: usize = HI_U64 as usize;

/// Returns `true` if `x` contains any zero byte.
///
/// From *Matters Computational*, J. Arndt:
///
/// "The idea is to subtract one from each of the bytes and then look for
/// bytes where the borrow propagated all the way to the most significant
/// bit."
#[inline]
fn contains_zero_byte(x: usize) -> bool {
    x.wrapping_sub(LO_USIZE) & !x & HI_USIZE != 0
}

#[cfg(target_pointer_width = "16")]
#[inline]
fn repeat_byte(b: u8) -> usize {
    (b as usize) << 8 | b as usize
}

#[cfg(not(target_pointer_width = "16"))]
#[inline]
fn repeat_byte(b: u8) -> usize {
    (b as usize) * (crate::usize::MAX / 255)
}

/// Returns the first index matching the byte `x` in `text`.
pub fn memchr(x: u8, text: &[u8]) -> Option<usize> {
    // Scan for a single byte value by reading two `usize` words at a time.
    //
    // Split `text` in three parts
    // - unaligned initial part, before the first word aligned address in text
    // - body, scan by 2 words at a time
    // - the last remaining part, < 2 word size
    let len = text.len();
    let ptr = text.as_ptr();
    let usize_bytes = mem::size_of::<usize>();

    // search up to an aligned boundary
    let mut offset = ptr.align_offset(usize_bytes);
    if offset > 0 {
        offset = cmp::min(offset, len);
        if let Some(index) = text[..offset].iter().position(|elt| *elt == x) {
            return Some(index);
        }
    }

    // search the body of the text
    let repeated_x = repeat_byte(x);

    if len >= 2 * usize_bytes {
        while offset <= len - 2 * usize_bytes {
            unsafe {
                let u = *(ptr.add(offset) as *const usize);
                let v = *(ptr.add(offset + usize_bytes) as *const usize);

                // break if there is a matching byte
                let zu = contains_zero_byte(u ^ repeated_x);
                let zv = contains_zero_byte(v ^ repeated_x);
                if zu || zv {
                    break;
                }
            }
            offset += usize_bytes * 2;
        }
    }

    // Find the byte after the point the body loop stopped.
    text[offset..].iter().position(|elt| *elt == x).map(|i| offset + i)
}

/// Returns the last index matching the byte `x` in `text`.
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
    let repeated_x = repeat_byte(x);
    let chunk_bytes = mem::size_of::<Chunk>();

    while offset > min_aligned_offset {
        unsafe {
            let u = *(ptr.offset(offset as isize - 2 * chunk_bytes as isize) as *const Chunk);
            let v = *(ptr.offset(offset as isize - chunk_bytes as isize) as *const Chunk);

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
