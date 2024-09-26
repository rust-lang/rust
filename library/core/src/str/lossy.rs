use super::from_utf8_unchecked;
use super::validations::utf8_char_width;
use crate::fmt;
use crate::fmt::{Formatter, Write};
use crate::iter::FusedIterator;

impl [u8] {
    /// Creates an iterator over the contiguous valid UTF-8 ranges of this
    /// slice, and the non-UTF-8 fragments in between.
    ///
    /// See the [`Utf8Chunk`] type for documenation of the items yielded by this iterator.
    ///
    /// # Examples
    ///
    /// This function formats arbitrary but mostly-UTF-8 bytes into Rust source
    /// code in the form of a C-string literal (`c"..."`).
    ///
    /// ```
    /// use std::fmt::Write as _;
    ///
    /// pub fn cstr_literal(bytes: &[u8]) -> String {
    ///     let mut repr = String::new();
    ///     repr.push_str("c\"");
    ///     for chunk in bytes.utf8_chunks() {
    ///         for ch in chunk.valid().chars() {
    ///             // Escapes \0, \t, \r, \n, \\, \', \", and uses \u{...} for non-printable characters.
    ///             write!(repr, "{}", ch.escape_debug()).unwrap();
    ///         }
    ///         for byte in chunk.invalid() {
    ///             write!(repr, "\\x{:02X}", byte).unwrap();
    ///         }
    ///     }
    ///     repr.push('"');
    ///     repr
    /// }
    ///
    /// fn main() {
    ///     let lit = cstr_literal(b"\xferris the \xf0\x9f\xa6\x80\x07");
    ///     let expected = stringify!(c"\xFErris the 🦀\u{7}");
    ///     assert_eq!(lit, expected);
    /// }
    /// ```
    #[stable(feature = "utf8_chunks", since = "1.79.0")]
    pub fn utf8_chunks(&self) -> Utf8Chunks<'_> {
        Utf8Chunks { source: self }
    }
}

/// An item returned by the [`Utf8Chunks`] iterator.
///
/// A `Utf8Chunk` stores a sequence of [`u8`] up to the first broken character
/// when decoding a UTF-8 string.
///
/// # Examples
///
/// ```
/// // An invalid UTF-8 string
/// let bytes = b"foo\xF1\x80bar";
///
/// // Decode the first `Utf8Chunk`
/// let chunk = bytes.utf8_chunks().next().unwrap();
///
/// // The first three characters are valid UTF-8
/// assert_eq!("foo", chunk.valid());
///
/// // The fourth character is broken
/// assert_eq!(b"\xF1\x80", chunk.invalid());
/// ```
#[stable(feature = "utf8_chunks", since = "1.79.0")]
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
    #[stable(feature = "utf8_chunks", since = "1.79.0")]
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
    #[stable(feature = "utf8_chunks", since = "1.79.0")]
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

        for chunk in self.0.utf8_chunks() {
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
/// See the [`Utf8Chunk`] type for documenation of the items yielded by this iterator.
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
/// fn from_utf8_lossy<F>(input: &[u8], mut push: F) where F: FnMut(&str) {
///     for chunk in input.utf8_chunks() {
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
#[stable(feature = "utf8_chunks", since = "1.79.0")]
#[derive(Clone)]
pub struct Utf8Chunks<'a> {
    source: &'a [u8],
}

impl<'a> Utf8Chunks<'a> {
    #[doc(hidden)]
    #[unstable(feature = "str_internals", issue = "none")]
    pub fn debug(&self) -> Debug<'_> {
        Debug(self.source)
    }
}

#[stable(feature = "utf8_chunks", since = "1.79.0")]
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

#[stable(feature = "utf8_chunks", since = "1.79.0")]
impl FusedIterator for Utf8Chunks<'_> {}

#[stable(feature = "utf8_chunks", since = "1.79.0")]
impl fmt::Debug for Utf8Chunks<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Utf8Chunks").field("source", &self.debug()).finish()
    }
}
