use super::from_utf8_unchecked;
use super::validations::run_utf8_validation;
use crate::fmt;
use crate::fmt::{Formatter, Write};
use crate::iter::FusedIterator;

impl [u8] {
    /// Creates an iterator over the contiguous valid UTF-8 ranges of this
    /// slice, and the non-UTF-8 fragments in between.
    ///
    /// See the [`Utf8Chunk`] type for documentation of the items yielded by this iterator.
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
/// This struct is created by the [`utf8_chunks`] method on bytes slices.
/// If you want a simple conversion from UTF-8 byte slices to string slices,
/// [`from_utf8`] is easier to use.
///
/// See the [`Utf8Chunk`] type for documentation of the items yielded by this iterator.
///
/// [byteslice]: slice
/// [`utf8_chunks`]: slice::utf8_chunks
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

        match run_utf8_validation(self.source) {
            Ok(()) => {
                // SAFETY: The whole `source` is valid in UTF-8.
                let valid = unsafe { from_utf8_unchecked(&self.source) };
                // Truncate the slice, no need to touch the pointer.
                self.source = &self.source[..0];
                Some(Utf8Chunk { valid, invalid: &[] })
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                let error_len = err.error_len().unwrap_or(self.source.len() - valid_up_to);
                // SAFETY: `valid_up_to` is the valid UTF-8 string length, so is in bound.
                let (valid, remaining) = unsafe { self.source.split_at_unchecked(valid_up_to) };
                // SAFETY: `error_len` is the errornous byte sequence length, so is in bound.
                let (invalid, after_invalid) = unsafe { remaining.split_at_unchecked(error_len) };
                self.source = after_invalid;
                Some(Utf8Chunk {
                    // SAFETY: All bytes up to `valid_up_to` are valid UTF-8.
                    valid: unsafe { from_utf8_unchecked(valid) },
                    invalid,
                })
            }
        }
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
