use crate::fmt;
use crate::fmt::Formatter;
use crate::fmt::Write;
use crate::iter::FusedIterator;

use super::from_utf8_unchecked;
use super::validations::utf8_char_width;

/// An item returned by the [`Utf8Chunks`] iterator.
///
/// A `Utf8Chunk` stores a sequence of [`u8`] up to the first broken character
/// when decoding a UTF-8 string.
///
/// # Examples
///
/// ```
/// #![feature(utf8_chunks)]
///
/// use std::str::Utf8Chunks;
///
/// // An invalid UTF-8 string
/// let bytes = b"foo\xF1\x80bar";
///
/// // Decode the first `Utf8Chunk`
/// let chunk = Utf8Chunks::new(bytes).next().unwrap();
///
/// // The first three characters are valid UTF-8
/// assert_eq!("foo", chunk.valid());
///
/// // The fourth character is broken
/// assert_eq!(b"\xF1\x80", chunk.invalid());
/// ```
#[unstable(feature = "utf8_chunks", issue = "99543")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Utf8Chunk<'a> {
    valid: &'a str,
    invalid: &'a [u8],
}

impl<'a> Utf8Chunk<'a> {
    /// Returns the next validated UTF-8 substring.
    ///
    /// This substring can be empty at the start of the string or between
    /// broken UTF-8 characters.
    #[must_use]
    #[unstable(feature = "utf8_chunks", issue = "99543")]
    pub fn valid(&self) -> &'a str {
        self.valid
    }

    /// Returns the invalid sequence that caused a failure.
    ///
    /// The returned slice will have a maximum length of 3 and starts after the
    /// substring given by [`valid`]. Decoding will resume after this sequence.
    ///
    /// If empty, this is the last chunk in the string. If non-empty, an
    /// unexpected byte was encountered or the end of the input was reached
    /// unexpectedly.
    ///
    /// Lossy decoding would replace this sequence with [`U+FFFD REPLACEMENT
    /// CHARACTER`].
    ///
    /// [`valid`]: Self::valid
    /// [`U+FFFD REPLACEMENT CHARACTER`]: crate::char::REPLACEMENT_CHARACTER
    #[must_use]
    #[unstable(feature = "utf8_chunks", issue = "99543")]
    pub fn invalid(&self) -> &'a [u8] {
        self.invalid
    }
}

#[must_use]
#[unstable(feature = "str_internals", issue = "none")]
pub struct Debug<'a>(&'a [u8]);

#[unstable(feature = "str_internals", issue = "none")]
impl fmt::Debug for Debug<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char('"')?;

        for chunk in Utf8Chunks::new(self.0) {
            // Valid part.
            // Here we partially parse UTF-8 again which is suboptimal.
            {
                let valid = chunk.valid();
                let mut from = 0;
                for (i, c) in valid.char_indices() {
                    let esc = c.escape_debug();
                    // If char needs escaping, flush backlog so far and write, else skip
                    if esc.len() != 1 {
                        f.write_str(&valid[from..i])?;
                        for c in esc {
                            f.write_char(c)?;
                        }
                        from = i + c.len_utf8();
                    }
                }
                f.write_str(&valid[from..])?;
            }

            // Broken parts of string as hex escape.
            for &b in chunk.invalid() {
                write!(f, "\\x{:02X}", b)?;
            }
        }

        f.write_char('"')
    }
}

/// An iterator used to decode a slice of mostly UTF-8 bytes to string slices
/// ([`&str`]) and byte slices ([`&[u8]`][byteslice]).
///
/// If you want a simple conversion from UTF-8 byte slices to string slices,
/// [`from_utf8`] is easier to use.
///
/// [byteslice]: slice
/// [`from_utf8`]: super::from_utf8
///
/// # Examples
///
/// This can be used to create functionality similar to
/// [`String::from_utf8_lossy`] without allocating heap memory:
///
/// ```
/// #![feature(utf8_chunks)]
///
/// use std::str::Utf8Chunks;
///
/// fn from_utf8_lossy<F>(input: &[u8], mut push: F) where F: FnMut(&str) {
///     for chunk in Utf8Chunks::new(input) {
///         push(chunk.valid());
///
///         if !chunk.invalid().is_empty() {
///             push("\u{FFFD}");
///         }
///     }
/// }
/// ```
///
/// [`String::from_utf8_lossy`]: ../../std/string/struct.String.html#method.from_utf8_lossy
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "utf8_chunks", issue = "99543")]
#[derive(Clone)]
pub struct Utf8Chunks<'a> {
    source: &'a [u8],
}

impl<'a> Utf8Chunks<'a> {
    /// Creates a new iterator to decode the bytes.
    #[unstable(feature = "utf8_chunks", issue = "99543")]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { source: bytes }
    }

    #[doc(hidden)]
    #[unstable(feature = "str_internals", issue = "none")]
    pub fn debug(&self) -> Debug<'_> {
        Debug(self.source)
    }
}

#[unstable(feature = "utf8_chunks", issue = "99543")]
impl<'a> Iterator for Utf8Chunks<'a> {
    type Item = Utf8Chunk<'a>;

    fn next(&mut self) -> Option<Utf8Chunk<'a>> {
        if self.source.is_empty() {
            return None;
        }

        const TAG_CONT_U8: u8 = 128;
        fn safe_get(xs: &[u8], i: usize) -> u8 {
            *xs.get(i).unwrap_or(&0)
        }

        let mut i = 0;
        let mut valid_up_to = 0;
        while i < self.source.len() {
            // SAFETY: `i < self.source.len()` per previous line.
            // For some reason the following are both significantly slower:
            // while let Some(&byte) = self.source.get(i) {
            // while let Some(byte) = self.source.get(i).copied() {
            let byte = unsafe { *self.source.get_unchecked(i) };
            i += 1;

            if byte < 128 {
                // This could be a `1 => ...` case in the match below, but for
                // the common case of all-ASCII inputs, we bypass loading the
                // sizeable UTF8_CHAR_WIDTH table into cache.
            } else {
                let w = utf8_char_width(byte);

                match w {
                    2 => {
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                    }
                    3 => {
                        match (byte, safe_get(self.source, i)) {
                            (0xE0, 0xA0..=0xBF) => (),
                            (0xE1..=0xEC, 0x80..=0xBF) => (),
                            (0xED, 0x80..=0x9F) => (),
                            (0xEE..=0xEF, 0x80..=0xBF) => (),
                            _ => break,
                        }
                        i += 1;
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                    }
                    4 => {
                        match (byte, safe_get(self.source, i)) {
                            (0xF0, 0x90..=0xBF) => (),
                            (0xF1..=0xF3, 0x80..=0xBF) => (),
                            (0xF4, 0x80..=0x8F) => (),
                            _ => break,
                        }
                        i += 1;
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                    }
                    _ => break,
                }
            }

            valid_up_to = i;
        }

        // SAFETY: `i <= self.source.len()` because it is only ever incremented
        // via `i += 1` and in between every single one of those increments, `i`
        // is compared against `self.source.len()`. That happens either
        // literally by `i < self.source.len()` in the while-loop's condition,
        // or indirectly by `safe_get(self.source, i) & 192 != TAG_CONT_U8`. The
        // loop is terminated as soon as the latest `i += 1` has made `i` no
        // longer less than `self.source.len()`, which means it'll be at most
        // equal to `self.source.len()`.
        let (inspected, remaining) = unsafe { self.source.split_at_unchecked(i) };
        self.source = remaining;

        // SAFETY: `valid_up_to <= i` because it is only ever assigned via
        // `valid_up_to = i` and `i` only increases.
        let (valid, invalid) = unsafe { inspected.split_at_unchecked(valid_up_to) };

        Some(Utf8Chunk {
            // SAFETY: All bytes up to `valid_up_to` are valid UTF-8.
            valid: unsafe { from_utf8_unchecked(valid) },
            invalid,
        })
    }
}

#[unstable(feature = "utf8_chunks", issue = "99543")]
impl FusedIterator for Utf8Chunks<'_> {}

#[unstable(feature = "utf8_chunks", issue = "99543")]
impl fmt::Debug for Utf8Chunks<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Utf8Chunks").field("source", &self.debug()).finish()
    }
}
