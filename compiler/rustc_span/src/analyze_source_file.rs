use super::*;
use unicode_width::UnicodeWidthChar;

#[cfg(test)]
mod tests;

/// Finds all newlines, multi-byte characters, and non-narrow characters in a
/// SourceFile.
///
/// This function will use an SSE2 enhanced implementation if hardware support
/// is detected at runtime.
pub fn analyze_source_file(
    src: &str,
    source_file_start_pos: BytePos,
) -> (Vec<BytePos>, Vec<MultiByteChar>, Vec<NonNarrowChar>) {
    let mut lines = vec![source_file_start_pos];
    let mut multi_byte_chars = vec![];
    let mut non_narrow_chars = vec![];

    // Calls the right implementation, depending on hardware support available.
    analyze_source_file_dispatch(
        src,
        source_file_start_pos,
        &mut lines,
        &mut multi_byte_chars,
        &mut non_narrow_chars,
    );

    // The code above optimistically registers a new line *after* each \n
    // it encounters. If that point is already outside the source_file, remove
    // it again.
    if let Some(&last_line_start) = lines.last() {
        let source_file_end = source_file_start_pos + BytePos::from_usize(src.len());
        assert!(source_file_end >= last_line_start);
        if last_line_start == source_file_end {
            lines.pop();
        }
    }

    (lines, multi_byte_chars, non_narrow_chars)
}

cfg_if::cfg_if! {
    if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64")))] {
        fn analyze_source_file_dispatch(src: &str,
                                    source_file_start_pos: BytePos,
                                    lines: &mut Vec<BytePos>,
                                    multi_byte_chars: &mut Vec<MultiByteChar>,
                                    non_narrow_chars: &mut Vec<NonNarrowChar>) {
            if is_x86_feature_detected!("sse2") {
                unsafe {
                    analyze_source_file_sse2(src,
                                         source_file_start_pos,
                                         lines,
                                         multi_byte_chars,
                                         non_narrow_chars);
                }
            } else {
                analyze_source_file_generic(src,
                                        src.len(),
                                        source_file_start_pos,
                                        lines,
                                        multi_byte_chars,
                                        non_narrow_chars);

            }
        }

        /// Checks 16 byte chunks of text at a time. If the chunk contains
        /// something other than printable ASCII characters and newlines, the
        /// function falls back to the generic implementation. Otherwise it uses
        /// SSE2 intrinsics to quickly find all newlines.
        #[target_feature(enable = "sse2")]
        unsafe fn analyze_source_file_sse2(src: &str,
                                       output_offset: BytePos,
                                       lines: &mut Vec<BytePos>,
                                       multi_byte_chars: &mut Vec<MultiByteChar>,
                                       non_narrow_chars: &mut Vec<NonNarrowChar>) {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;

            const CHUNK_SIZE: usize = 16;

            let src_bytes = src.as_bytes();

            let chunk_count = src.len() / CHUNK_SIZE;

            // This variable keeps track of where we should start decoding a
            // chunk. If a multi-byte character spans across chunk boundaries,
            // we need to skip that part in the next chunk because we already
            // handled it.
            let mut intra_chunk_offset = 0;

            for chunk_index in 0 .. chunk_count {
                let ptr = src_bytes.as_ptr() as *const __m128i;
                // We don't know if the pointer is aligned to 16 bytes, so we
                // use `loadu`, which supports unaligned loading.
                let chunk = _mm_loadu_si128(ptr.add(chunk_index));

                // For character in the chunk, see if its byte value is < 0, which
                // indicates that it's part of a UTF-8 char.
                let multibyte_test = _mm_cmplt_epi8(chunk, _mm_set1_epi8(0));
                // Create a bit mask from the comparison results.
                let multibyte_mask = _mm_movemask_epi8(multibyte_test);

                // If the bit mask is all zero, we only have ASCII chars here:
                if multibyte_mask == 0 {
                    assert!(intra_chunk_offset == 0);

                    // Check if there are any control characters in the chunk. All
                    // control characters that we can encounter at this point have a
                    // byte value less than 32 or ...
                    let control_char_test0 = _mm_cmplt_epi8(chunk, _mm_set1_epi8(32));
                    let control_char_mask0 = _mm_movemask_epi8(control_char_test0);

                    // ... it's the ASCII 'DEL' character with a value of 127.
                    let control_char_test1 = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(127));
                    let control_char_mask1 = _mm_movemask_epi8(control_char_test1);

                    let control_char_mask = control_char_mask0 | control_char_mask1;

                    if control_char_mask != 0 {
                        // Check for newlines in the chunk
                        let newlines_test = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\n' as i8));
                        let newlines_mask = _mm_movemask_epi8(newlines_test);

                        if control_char_mask == newlines_mask {
                            // All control characters are newlines, record them
                            let mut newlines_mask = 0xFFFF0000 | newlines_mask as u32;
                            let output_offset = output_offset +
                                BytePos::from_usize(chunk_index * CHUNK_SIZE + 1);

                            loop {
                                let index = newlines_mask.trailing_zeros();

                                if index >= CHUNK_SIZE as u32 {
                                    // We have arrived at the end of the chunk.
                                    break
                                }

                                lines.push(BytePos(index) + output_offset);

                                // Clear the bit, so we can find the next one.
                                newlines_mask &= (!1) << index;
                            }

                            // We are done for this chunk. All control characters were
                            // newlines and we took care of those.
                            continue
                        } else {
                            // Some of the control characters are not newlines,
                            // fall through to the slow path below.
                        }
                    } else {
                        // No control characters, nothing to record for this chunk
                        continue
                    }
                }

                // The slow path.
                // There are control chars in here, fallback to generic decoding.
                let scan_start = chunk_index * CHUNK_SIZE + intra_chunk_offset;
                intra_chunk_offset = analyze_source_file_generic(
                    &src[scan_start .. ],
                    CHUNK_SIZE - intra_chunk_offset,
                    BytePos::from_usize(scan_start) + output_offset,
                    lines,
                    multi_byte_chars,
                    non_narrow_chars
                );
            }

            // There might still be a tail left to analyze
            let tail_start = chunk_count * CHUNK_SIZE + intra_chunk_offset;
            if tail_start < src.len() {
                analyze_source_file_generic(&src[tail_start as usize ..],
                                        src.len() - tail_start,
                                        output_offset + BytePos::from_usize(tail_start),
                                        lines,
                                        multi_byte_chars,
                                        non_narrow_chars);
            }
        }
    } else {

        // The target (or compiler version) does not support SSE2 ...
        fn analyze_source_file_dispatch(src: &str,
                                    source_file_start_pos: BytePos,
                                    lines: &mut Vec<BytePos>,
                                    multi_byte_chars: &mut Vec<MultiByteChar>,
                                    non_narrow_chars: &mut Vec<NonNarrowChar>) {
            analyze_source_file_generic(src,
                                    src.len(),
                                    source_file_start_pos,
                                    lines,
                                    multi_byte_chars,
                                    non_narrow_chars);
        }
    }
}

// `scan_len` determines the number of bytes in `src` to scan. Note that the
// function can read past `scan_len` if a multi-byte character start within the
// range but extends past it. The overflow is returned by the function.
fn analyze_source_file_generic(
    src: &str,
    scan_len: usize,
    output_offset: BytePos,
    lines: &mut Vec<BytePos>,
    multi_byte_chars: &mut Vec<MultiByteChar>,
    non_narrow_chars: &mut Vec<NonNarrowChar>,
) -> usize {
    assert!(src.len() >= scan_len);
    let mut i = 0;
    let src_bytes = src.as_bytes();

    while i < scan_len {
        let byte = unsafe {
            // We verified that i < scan_len <= src.len()
            *src_bytes.get_unchecked(i as usize)
        };

        // How much to advance in order to get to the next UTF-8 char in the
        // string.
        let mut char_len = 1;

        if byte < 32 {
            // This is an ASCII control character, it could be one of the cases
            // that are interesting to us.

            let pos = BytePos::from_usize(i) + output_offset;

            match byte {
                b'\n' => {
                    lines.push(pos + BytePos(1));
                }
                b'\t' => {
                    non_narrow_chars.push(NonNarrowChar::Tab(pos));
                }
                _ => {
                    non_narrow_chars.push(NonNarrowChar::ZeroWidth(pos));
                }
            }
        } else if byte >= 127 {
            // The slow path:
            // This is either ASCII control character "DEL" or the beginning of
            // a multibyte char. Just decode to `char`.
            let c = (&src[i..]).chars().next().unwrap();
            char_len = c.len_utf8();

            let pos = BytePos::from_usize(i) + output_offset;

            if char_len > 1 {
                assert!((2..=4).contains(&char_len));
                let mbc = MultiByteChar { pos, bytes: char_len as u8 };
                multi_byte_chars.push(mbc);
            }

            // Assume control characters are zero width.
            // FIXME: How can we decide between `width` and `width_cjk`?
            let char_width = UnicodeWidthChar::width(c).unwrap_or(0);

            if char_width != 1 {
                non_narrow_chars.push(NonNarrowChar::new(pos, char_width));
            }
        }

        i += char_len;
    }

    i - scan_len
}
