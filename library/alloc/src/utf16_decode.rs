use core::mem::MaybeUninit;

use crate::vec::Vec;

const BLOCK: usize = 16;
// Three bytes per unit, plus one for the last store or a pair crossing the block.
const ENCODE_BLOCK_LEN: usize = 3 * BLOCK + 1;

/// Appends generalized UTF-8; `on_unpaired` handles lone surrogates.
/// Callers must preserve their encoding invariant. On error, `out` is unspecified.
pub(crate) fn decode_utf16_into(
    v: &[u16],
    out: &mut Vec<u8>,
    mut on_unpaired: impl FnMut(u16) -> Result<u16, ()>,
) -> Result<(), ()> {
    let mut i = 0;
    while let Some(units) = v.get(i..i + BLOCK) {
        let units: &[u16; BLOCK] = units.try_into().unwrap();
        let mut max = 0;
        let mut surrogates = false;
        for &u in units {
            max = max.max(u);
            surrogates |= (u & 0xF800) == 0xD800;
        }
        let decoded_len = if max < 0x80 {
            let rest = &v[i..];
            out.reserve(rest.len());

            let ascii_len = ascii_prefix(rest, out.spare_capacity_mut());
            // SAFETY: `ascii_prefix` initialized `ascii_len` bytes in the reserved space.
            unsafe { out.set_len(out.len() + ascii_len) };
            ascii_len
        } else if surrogates {
            out.reserve(ENCODE_BLOCK_LEN);
            // SAFETY: We reserved `3 * BLOCK + 1` bytes.
            let (len, written) = unsafe {
                decode_scalar(v, i, i + BLOCK, out.spare_capacity_mut(), &mut on_unpaired)
            }?;
            // SAFETY: `decode_scalar` initialized `written` bytes in the reserved space.
            unsafe { out.set_len(out.len() + written) };
            len
        } else {
            out.reserve(v.len() - i + ENCODE_BLOCK_LEN);
            let len = encode_block(units, out.spare_capacity_mut());
            // SAFETY: `encode_block` initialized `len` bytes in the reserved space.
            unsafe {
                out.set_len(out.len() + len);
            };
            BLOCK
        };
        i += decoded_len;
    }
    decode_small(&v[i..], out, &mut on_unpaired)?;
    Ok(())
}

fn decode_small(
    v: &[u16],
    out: &mut Vec<u8>,
    on_unpaired: &mut impl FnMut(u16) -> Result<u16, ()>,
) -> Result<(), ()> {
    let mut i = 0;
    while i < v.len() {
        let u = v[i];
        i += 1;
        let c = if u < 0x80 {
            out.push(u as u8);
            continue;
        } else if (u & 0xF800) != 0xD800 {
            u as u32
        } else if u < 0xDC00 && i < v.len() && (v[i] & 0xFC00) == 0xDC00 {
            let c = 0x10000 + (((u as u32) & 0x3FF) << 10) + ((v[i] as u32) & 0x3FF);
            i += 1;
            c
        } else {
            on_unpaired(u)? as u32
        };
        push_code_point(out, c);
    }
    Ok(())
}

#[inline]
fn push_code_point(out: &mut Vec<u8>, code: u32) {
    let n = 1 + (code >= 0x80) as usize + (code >= 0x800) as usize + (code >= 0x10000) as usize;
    let len = out.len();
    out.reserve(n);

    // SAFETY: We reserved `n` bytes, and the encoder initializes all of them.
    unsafe {
        core::char::encode_utf8_raw_unchecked(code, out.as_mut_ptr().add(len));
        out.set_len(len + n);
    }
}

fn ascii_prefix(src: &[u16], out: &mut [MaybeUninit<u8>]) -> usize {
    const CHUNK_SIZE: usize = 32;
    let mut i = 0;
    while let Some(units) = src.get(i..i + CHUNK_SIZE) {
        let mut max: u16 = 0;
        let units: &[u16; CHUNK_SIZE] = units.try_into().unwrap();
        for j in 0..CHUNK_SIZE {
            max = max.max(units[j]);
        }
        if max >= 0x80 {
            break;
        }
        let dst: &mut [MaybeUninit<u8>; CHUNK_SIZE] =
            (&mut out[i..i + CHUNK_SIZE]).try_into().unwrap();
        for j in 0..CHUNK_SIZE {
            dst[j].write(units[j] as u8);
        }
        i += CHUNK_SIZE;
    }
    while i < src.len() && src[i] < 0x80 {
        out[i].write(src[i] as u8);
        i += 1;
    }
    i
}

/// Decodes `v[start..end]`, returning `(units_consumed, bytes_written)`.
/// A pair crossing `end` consumes one extra unit.
///
/// # Safety
///
/// `out.len()` must be at least `3 * (end - start) + 1`.
unsafe fn decode_scalar(
    v: &[u16],
    start: usize,
    end: usize,
    out: &mut [MaybeUninit<u8>],
    on_unpaired: &mut impl FnMut(u16) -> Result<u16, ()>,
) -> Result<(usize, usize), ()> {
    let mut i = start;
    let mut written = 0;
    while i < end {
        let u = v[i];
        i += 1;
        let (word, n) = if u < 0x80 {
            (u as u32, 1)
        } else if (u & 0xF800) != 0xD800 {
            encode_bmp(u)
        } else if u < 0xDC00 && i < v.len() && (v[i] & 0xFC00) == 0xDC00 {
            let c = 0x10000 + (((u as u32) & 0x3FF) << 10) + ((v[i] as u32) & 0x3FF);
            let cont = |shift: u32| 0x80 | (c >> shift & 0x3F);
            i += 1;
            ((0xF0 | c >> 18) | cont(12) << 8 | cont(6) << 16 | cont(0) << 24, 4)
        } else {
            let Ok(unit) = on_unpaired(u) else {
                return Err(());
            };
            encode_bmp(unit)
        };

        // SAFETY: The caller's output bound covers the final store and a pair crossing `end`.
        unsafe {
            let dst = out.get_unchecked_mut(written..).as_mut_ptr();
            match n {
                1 => dst.cast::<u8>().write(word as u8),
                2 => dst.cast::<u16>().write_unaligned((word as u16).to_le()),
                3 | 4 => dst.cast::<u32>().write_unaligned(word.to_le()),
                _ => unreachable!(),
            }
        }
        written += n;
    }
    Ok((i - start, written))
}

/// Packs UTF-8 bytes least-significant byte first, and returns their length.
/// Surrogates are encoded as-is.
#[inline]
fn encode_bmp(u: u16) -> (u32, usize) {
    let u = u as u32;
    if u < 0x80 {
        (u, 1)
    } else if u < 0x800 {
        ((0xC0 | u >> 6) | (0x80 | u & 0x3F) << 8, 2)
    } else {
        ((0xE0 | u >> 12) | (0x80 | (u >> 6) & 0x3F) << 8 | (0x80 | u & 0x3F) << 16, 3)
    }
}

/// Encodes a non-surrogate block and returns the byte count.
fn encode_block(units: &[u16; BLOCK], out: &mut [MaybeUninit<u8>]) -> usize {
    // Byte lengths keep the encoding loop vectorizable.
    let mut words = [0u32; BLOCK];
    let mut lens = [0u8; BLOCK];
    // Encoding separately from the stores lets LLVM vectorize the first loop.
    for k in 0..BLOCK {
        let (word, n) = encode_bmp(units[k]);
        (words[k], lens[k]) = (word, n as u8);
    }
    let mut len = 0;
    for k in 0..BLOCK {
        // Four-byte stores overlap; the last may extend past the returned count.
        out[len..len + 4].write_copy_of_slice(&words[k].to_le_bytes());
        len += lens[k] as usize;
    }
    len
}
