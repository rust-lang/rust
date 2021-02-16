//! Enhances `ide::LineIndex` with additional info required to convert offsets
//! into lsp positions.
//!
//! We maintain invariant that all internal strings use `\n` as line separator.
//! This module does line ending conversion and detection (so that we can
//! convert back to `\r\n` on the way out).

use std::sync::Arc;

pub enum OffsetEncoding {
    Utf8,
    Utf16,
}

pub(crate) struct LineIndex {
    pub(crate) index: Arc<ide::LineIndex>,
    pub(crate) endings: LineEndings,
    pub(crate) encoding: OffsetEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum LineEndings {
    Unix,
    Dos,
}

impl LineEndings {
    /// Replaces `\r\n` with `\n` in-place in `src`.
    pub(crate) fn normalize(src: String) -> (String, LineEndings) {
        if !src.as_bytes().contains(&b'\r') {
            return (src, LineEndings::Unix);
        }

        // We replace `\r\n` with `\n` in-place, which doesn't break utf-8 encoding.
        // While we *can* call `as_mut_vec` and do surgery on the live string
        // directly, let's rather steal the contents of `src`. This makes the code
        // safe even if a panic occurs.

        let mut buf = src.into_bytes();
        let mut gap_len = 0;
        let mut tail = buf.as_mut_slice();
        loop {
            let idx = match find_crlf(&tail[gap_len..]) {
                None => tail.len(),
                Some(idx) => idx + gap_len,
            };
            tail.copy_within(gap_len..idx, 0);
            tail = &mut tail[idx - gap_len..];
            if tail.len() == gap_len {
                break;
            }
            gap_len += 1;
        }

        // Account for removed `\r`.
        // After `set_len`, `buf` is guaranteed to contain utf-8 again.
        let new_len = buf.len() - gap_len;
        let src = unsafe {
            buf.set_len(new_len);
            String::from_utf8_unchecked(buf)
        };
        return (src, LineEndings::Dos);

        fn find_crlf(src: &[u8]) -> Option<usize> {
            src.windows(2).position(|it| it == b"\r\n")
        }
    }
}
