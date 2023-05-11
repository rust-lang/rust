//! Enhances `ide::LineIndex` with additional info required to convert offsets
//! into lsp positions.
//!
//! We maintain invariant that all internal strings use `\n` as line separator.
//! This module does line ending conversion and detection (so that we can
//! convert back to `\r\n` on the way out).

use std::sync::Arc;

use ide_db::line_index::WideEncoding;

#[derive(Clone, Copy)]
pub enum PositionEncoding {
    Utf8,
    Wide(WideEncoding),
}

pub(crate) struct LineIndex {
    pub(crate) index: Arc<ide::LineIndex>,
    pub(crate) endings: LineEndings,
    pub(crate) encoding: PositionEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum LineEndings {
    Unix,
    Dos,
}

impl LineEndings {
    /// Replaces `\r\n` with `\n` in-place in `src`.
    pub(crate) fn normalize(src: String) -> (String, LineEndings) {
        // We replace `\r\n` with `\n` in-place, which doesn't break utf-8 encoding.
        // While we *can* call `as_mut_vec` and do surgery on the live string
        // directly, let's rather steal the contents of `src`. This makes the code
        // safe even if a panic occurs.

        let mut buf = src.into_bytes();
        let mut gap_len = 0;
        let mut tail = buf.as_mut_slice();
        let mut crlf_seen = false;

        let find_crlf = |src: &[u8]| src.windows(2).position(|it| it == b"\r\n");

        loop {
            let idx = match find_crlf(&tail[gap_len..]) {
                None if crlf_seen => tail.len(),
                // SAFETY: buf is unchanged and therefore still contains utf8 data
                None => return (unsafe { String::from_utf8_unchecked(buf) }, LineEndings::Unix),
                Some(idx) => {
                    crlf_seen = true;
                    idx + gap_len
                }
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
        let src = unsafe {
            let new_len = buf.len() - gap_len;
            buf.set_len(new_len);
            String::from_utf8_unchecked(buf)
        };
        (src, LineEndings::Dos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix() {
        let src = "a\nb\nc\n\n\n\n";
        let (res, endings) = LineEndings::normalize(src.into());
        assert_eq!(endings, LineEndings::Unix);
        assert_eq!(res, src);
    }

    #[test]
    fn dos() {
        let src = "\r\na\r\n\r\nb\r\nc\r\n\r\n\r\n\r\n";
        let (res, endings) = LineEndings::normalize(src.into());
        assert_eq!(endings, LineEndings::Dos);
        assert_eq!(res, "\na\n\nb\nc\n\n\n\n");
    }

    #[test]
    fn mixed() {
        let src = "a\r\nb\r\nc\r\n\n\r\n\n";
        let (res, endings) = LineEndings::normalize(src.into());
        assert_eq!(endings, LineEndings::Dos);
        assert_eq!(res, "a\nb\nc\n\n\n\n");
    }

    #[test]
    fn none() {
        let src = "abc";
        let (res, endings) = LineEndings::normalize(src.into());
        assert_eq!(endings, LineEndings::Unix);
        assert_eq!(res, src);
    }
}
