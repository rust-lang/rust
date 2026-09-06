// Original implementation taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

use crate::intrinsics::const_eval_select;

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
const LO_USIZE: usize = usize::repeat_u8(0x01);
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
const HI_USIZE: usize = usize::repeat_u8(0x80);
const USIZE_BYTES: usize = size_of::<usize>();

/// Returns `true` if `x` contains any zero byte.
///
/// From *Matters Computational*, J. Arndt:
///
/// "The idea is to subtract one from each of the bytes and then look for
/// bytes where the borrow propagated all the way to the most significant
/// bit."
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
const fn contains_zero_byte(x: usize) -> bool {
    x.wrapping_sub(LO_USIZE) & !x & HI_USIZE != 0
}

/// Returns the first index matching the byte `x` in `text`.
#[inline]
#[must_use]
pub const fn memchr(x: u8, text: &[u8]) -> Option<usize> {
    // Fast path for small slices.
    let result =
        if text.len() < 2 * USIZE_BYTES { memchr_naive(x, text) } else { memchr_wide(x, text) };
    if let Some(index) = result {
        // SAFETY: Both implementations only return an index from within `text`.
        unsafe { crate::hint::assert_unchecked(index < text.len()) };
    }
    result
}

#[inline]
const fn memchr_naive(x: u8, text: &[u8]) -> Option<usize> {
    let mut i = 0;

    // FIXME(const-hack): Replace with `text.iter().pos(|c| *c == x)`.
    while i < text.len() {
        if text[i] == x {
            return Some(i);
        }

        i += 1;
    }

    None
}

#[rustc_allow_const_fn_unstable(const_eval_select)] // fallback impl has same behavior
const fn memchr_wide(x: u8, text: &[u8]) -> Option<usize> {
    // The runtime version behaves the same as the compiletime version, it's
    // just more optimized.
    const_eval_select!(
        @capture { x: u8, text: &[u8] } -> Option<usize>:
        if const {
            memchr_naive(x, text)
        } else {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            { memchr_vectored(x, text) }
            #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
            { memchr_aligned(x, text) }
        }
    )
}

/// Scan for a single byte value by reading two `usize` words at a time.
///
/// Only called from the runtime arm of `memchr_wide`'s `const_eval_select`,
/// so it does not need to be (and cannot be: `align_offset`) a `const fn`.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
fn memchr_aligned(x: u8, text: &[u8]) -> Option<usize> {
    // Split `text` in three parts
    // - unaligned initial part, before the first word aligned address in text
    // - body, scan by 2 words at a time
    // - the last remaining part, < 2 word size

    // search up to an aligned boundary
    let len = text.len();
    let ptr = text.as_ptr();
    let mut offset = ptr.align_offset(USIZE_BYTES);

    if offset > 0 {
        offset = offset.min(len);
        let slice = &text[..offset];
        if let Some(index) = memchr_naive(x, slice) {
            return Some(index);
        }
    }

    // search the body of the text
    let repeated_x = usize::repeat_u8(x);
    while offset <= len - 2 * USIZE_BYTES {
        // SAFETY: the while's predicate guarantees a distance of at least 2 * usize_bytes
        // between the offset and the end of the slice.
        unsafe {
            let u = *(ptr.add(offset) as *const usize);
            let v = *(ptr.add(offset + USIZE_BYTES) as *const usize);

            // break if there is a matching byte
            let zu = contains_zero_byte(u ^ repeated_x);
            let zv = contains_zero_byte(v ^ repeated_x);
            if zu || zv {
                break;
            }
        }
        offset += USIZE_BYTES * 2;
    }

    // Find the byte after the point the body loop stopped.
    // LLVM cannot prove `offset <= len` here, so `&text[offset..]`
    // emits an unreachable  bounds-check panic branch.
    // SAFETY: offset is within bounds
    let slice = unsafe { super::from_raw_parts(ptr.add(offset), len - offset) };
    memchr_naive(x, slice).map(|i| offset + i)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const CHUNK_SIZE: usize = 64;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const VECTOR_SIZE: usize = 16;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn chunk_contains(x: u8, chunk: &[u8; CHUNK_SIZE]) -> bool {
    use crate::simd::cmp::SimdPartialEq;
    use crate::simd::u8x16;

    let needle = u8x16::splat(x);
    let (vectors, _) = chunk.as_chunks::<VECTOR_SIZE>();
    let any = u8x16::from_array(vectors[0]).simd_eq(needle)
        | u8x16::from_array(vectors[1]).simd_eq(needle)
        | u8x16::from_array(vectors[2]).simd_eq(needle)
        | u8x16::from_array(vectors[3]).simd_eq(needle);
    any.any()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn vector_contains(x: u8, vector: &[u8; VECTOR_SIZE]) -> bool {
    use crate::simd::cmp::SimdPartialEq;
    use crate::simd::u8x16;

    u8x16::from_array(*vector).simd_eq(u8x16::splat(x)).any()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn memchr_vectored(x: u8, text: &[u8]) -> Option<usize> {
    // Main loop: one any-match reduction per 64 bytes, so the long-latency
    // horizontal reduction runs once per four vectors rather than per load.
    let (chunks, _) = text.as_chunks::<CHUNK_SIZE>();
    let mut offset = 0;
    for chunk in chunks {
        if chunk_contains(x, chunk) {
            break;
        }
        offset += CHUNK_SIZE;
    }

    // SAFETY: `offset` advances by whole chunks of the `as_chunks`
    // decomposition, so `offset <= text.len()`.
    unsafe { crate::hint::assert_unchecked(offset <= text.len()) };

    // Single-vector steps: sweep the tail left by the unrolled loop, and
    // narrow a 64-byte hit down to the 16-byte block containing the match.
    let (vectors, _) = text[offset..].as_chunks::<VECTOR_SIZE>();
    for vector in vectors {
        if vector_contains(x, vector) {
            break;
        }
        offset += VECTOR_SIZE;
    }

    // SAFETY: as above, `offset` only advanced by whole vectors.
    unsafe { crate::hint::assert_unchecked(offset <= text.len()) };

    // Exact index recovery within the candidate 16-byte block, or the final
    // < 16-byte scalar tail.
    match text[offset..].iter().position(|&b| b == x) {
        Some(i) => Some(offset + i),
        None => None,
    }
}

/// Returns the last index matching the byte `x` in `text`.
#[inline]
#[must_use]
pub fn memrchr(x: u8, text: &[u8]) -> Option<usize> {
    let result = {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            memrchr_vectored(x, text)
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            memrchr_aligned(x, text)
        }
    };
    if let Some(index) = result {
        // SAFETY: every implementation only returns the index of a matching byte in `text`.
        unsafe { crate::hint::assert_unchecked(index < text.len()) };
    }
    result
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
fn memrchr_aligned(x: u8, text: &[u8]) -> Option<usize> {
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn memrchr_vectored(x: u8, text: &[u8]) -> Option<usize> {
    let (_, chunks) = text.as_rchunks::<CHUNK_SIZE>();
    let mut end = text.len();
    for chunk in chunks.iter().rev() {
        if chunk_contains(x, chunk) {
            break;
        }
        end -= CHUNK_SIZE;
    }

    // SAFETY: `end` decreases by whole chunks of the `as_rchunks`
    // decomposition, so `end <= text.len()`.
    unsafe { crate::hint::assert_unchecked(end <= text.len()) };

    let (_, vectors) = text[..end].as_rchunks::<VECTOR_SIZE>();
    for vector in vectors.iter().rev() {
        if vector_contains(x, vector) {
            break;
        }
        end -= VECTOR_SIZE;
    }

    // SAFETY: as above, `end` only decreased by whole vectors.
    unsafe { crate::hint::assert_unchecked(end <= text.len()) };

    text[..end].iter().rposition(|&b| b == x)
}
