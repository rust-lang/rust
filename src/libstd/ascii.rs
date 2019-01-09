//! Operations on ASCII strings and characters.
//!
//! Most string operations in Rust act on UTF-8 strings. However, at times it
//! makes more sense to only consider the ASCII character set for a specific
//! operation.
//!
//! The [`AsciiExt`] trait provides methods that allow for character
//! operations that only act on the ASCII subset and leave non-ASCII characters
//! alone.
//!
//! The [`escape_default`] function provides an iterator over the bytes of an
//! escaped version of the character given.
//!
//! [`AsciiExt`]: trait.AsciiExt.html
//! [`escape_default`]: fn.escape_default.html

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ascii::{EscapeDefault, escape_default};

/// Extension methods for ASCII-subset only operations.
///
/// Be aware that operations on seemingly non-ASCII characters can sometimes
/// have unexpected results. Consider this example:
///
/// ```
/// use std::ascii::AsciiExt;
///
/// assert_eq!(AsciiExt::to_ascii_uppercase("café"), "CAFÉ");
/// assert_eq!(AsciiExt::to_ascii_uppercase("café"), "CAFé");
/// ```
///
/// In the first example, the lowercased string is represented `"cafe\u{301}"`
/// (the last character is an acute accent [combining character]). Unlike the
/// other characters in the string, the combining character will not get mapped
/// to an uppercase variant, resulting in `"CAFE\u{301}"`. In the second
/// example, the lowercased string is represented `"caf\u{e9}"` (the last
/// character is a single Unicode character representing an 'e' with an acute
/// accent). Since the last character is defined outside the scope of ASCII,
/// it will not get mapped to an uppercase variant, resulting in `"CAF\u{e9}"`.
///
/// [combining character]: https://en.wikipedia.org/wiki/Combining_character
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(since = "1.26.0", reason = "use inherent methods instead")]
pub trait AsciiExt {
    /// Container type for copied ASCII characters.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Owned;

    /// Checks if the value is within the ASCII range.
    ///
    /// # Note
    ///
    /// This method will be deprecated in favor of the identically-named
    /// inherent methods on `u8`, `char`, `[u8]` and `str`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_ascii(&self) -> bool;

    /// Makes a copy of the value in its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// To uppercase ASCII characters in addition to non-ASCII characters, use
    /// [`str::to_uppercase`].
    ///
    /// # Note
    ///
    /// This method will be deprecated in favor of the identically-named
    /// inherent methods on `u8`, `char`, `[u8]` and `str`.
    ///
    /// [`make_ascii_uppercase`]: #tymethod.make_ascii_uppercase
    /// [`str::to_uppercase`]: ../primitive.str.html#method.to_uppercase
    #[stable(feature = "rust1", since = "1.0.0")]
    #[allow(deprecated)]
    fn to_ascii_uppercase(&self) -> Self::Owned;

    /// Makes a copy of the value in its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// To lowercase ASCII characters in addition to non-ASCII characters, use
    /// [`str::to_lowercase`].
    ///
    /// # Note
    ///
    /// This method will be deprecated in favor of the identically-named
    /// inherent methods on `u8`, `char`, `[u8]` and `str`.
    ///
    /// [`make_ascii_lowercase`]: #tymethod.make_ascii_lowercase
    /// [`str::to_lowercase`]: ../primitive.str.html#method.to_lowercase
    #[stable(feature = "rust1", since = "1.0.0")]
    #[allow(deprecated)]
    fn to_ascii_lowercase(&self) -> Self::Owned;

    /// Checks that two values are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    ///
    /// # Note
    ///
    /// This method will be deprecated in favor of the identically-named
    /// inherent methods on `u8`, `char`, `[u8]` and `str`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn eq_ignore_ascii_case(&self, other: &Self) -> bool;

    /// Converts this type to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// # Note
    ///
    /// This method will be deprecated in favor of the identically-named
    /// inherent methods on `u8`, `char`, `[u8]` and `str`.
    ///
    /// [`to_ascii_uppercase`]: #tymethod.to_ascii_uppercase
    #[stable(feature = "ascii", since = "1.9.0")]
    fn make_ascii_uppercase(&mut self);

    /// Converts this type to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// # Note
    ///
    /// This method will be deprecated in favor of the identically-named
    /// inherent methods on `u8`, `char`, `[u8]` and `str`.
    ///
    /// [`to_ascii_lowercase`]: #tymethod.to_ascii_lowercase
    #[stable(feature = "ascii", since = "1.9.0")]
    fn make_ascii_lowercase(&mut self);
}

macro_rules! delegating_ascii_methods {
    () => {
        #[inline]
        fn is_ascii(&self) -> bool { self.is_ascii() }

        #[inline]
        fn to_ascii_uppercase(&self) -> Self::Owned { self.to_ascii_uppercase() }

        #[inline]
        fn to_ascii_lowercase(&self) -> Self::Owned { self.to_ascii_lowercase() }

        #[inline]
        fn eq_ignore_ascii_case(&self, o: &Self) -> bool { self.eq_ignore_ascii_case(o) }

        #[inline]
        fn make_ascii_uppercase(&mut self) { self.make_ascii_uppercase(); }

        #[inline]
        fn make_ascii_lowercase(&mut self) { self.make_ascii_lowercase(); }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl AsciiExt for u8 {
    type Owned = u8;

    delegating_ascii_methods!();
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl AsciiExt for char {
    type Owned = char;

    delegating_ascii_methods!();
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl AsciiExt for [u8] {
    type Owned = Vec<u8>;

    delegating_ascii_methods!();
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl AsciiExt for str {
    type Owned = String;

    delegating_ascii_methods!();
}
