//! String manipulation.
//!
//! For more details, see the [`std::str`] module.
//!
//! [`std::str`]: ../../std/str/index.html

#![stable(feature = "rust1", since = "1.0.0")]

mod converts;
mod count;
mod error;
mod iter;
mod traits;
mod validations;

use self::pattern::{DoubleEndedSearcher, Pattern, ReverseSearcher, Searcher};
use crate::char::{self, EscapeDebugExtArgs};
use crate::ops::Range;
use crate::slice::{self, SliceIndex};
use crate::{ascii, mem};

pub mod pattern;

mod lossy;
#[unstable(feature = "str_from_raw_parts", issue = "119206")]
pub use converts::{from_raw_parts, from_raw_parts_mut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use converts::{from_utf8, from_utf8_unchecked};
#[stable(feature = "str_mut_extras", since = "1.20.0")]
pub use converts::{from_utf8_mut, from_utf8_unchecked_mut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use error::{ParseBoolError, Utf8Error};
#[stable(feature = "encode_utf16", since = "1.8.0")]
pub use iter::EncodeUtf16;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
pub use iter::LinesAny;
#[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
pub use iter::SplitAsciiWhitespace;
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub use iter::SplitInclusive;
#[stable(feature = "rust1", since = "1.0.0")]
pub use iter::{Bytes, CharIndices, Chars, Lines, SplitWhitespace};
#[stable(feature = "str_escape", since = "1.34.0")]
pub use iter::{EscapeDebug, EscapeDefault, EscapeUnicode};
#[stable(feature = "str_match_indices", since = "1.5.0")]
pub use iter::{MatchIndices, RMatchIndices};
use iter::{MatchIndicesInternal, MatchesInternal, SplitInternal, SplitNInternal};
#[stable(feature = "str_matches", since = "1.2.0")]
pub use iter::{Matches, RMatches};
#[stable(feature = "rust1", since = "1.0.0")]
pub use iter::{RSplit, RSplitTerminator, Split, SplitTerminator};
#[stable(feature = "rust1", since = "1.0.0")]
pub use iter::{RSplitN, SplitN};
#[stable(feature = "utf8_chunks", since = "1.79.0")]
pub use lossy::{Utf8Chunk, Utf8Chunks};
#[stable(feature = "rust1", since = "1.0.0")]
pub use traits::FromStr;
#[unstable(feature = "str_internals", issue = "none")]
pub use validations::{next_code_point, utf8_char_width};

#[inline(never)]
#[cold]
#[track_caller]
#[rustc_allow_const_fn_unstable(const_eval_select)]
#[cfg(not(feature = "panic_immediate_abort"))]
const fn slice_error_fail(s: &str, begin: usize, end: usize) -> ! {
    crate::intrinsics::const_eval_select((s, begin, end), slice_error_fail_ct, slice_error_fail_rt)
}

#[cfg(feature = "panic_immediate_abort")]
const fn slice_error_fail(s: &str, begin: usize, end: usize) -> ! {
    slice_error_fail_ct(s, begin, end)
}

#[track_caller]
const fn slice_error_fail_ct(_: &str, _: usize, _: usize) -> ! {
    panic!("failed to slice string");
}

#[track_caller]
fn slice_error_fail_rt(s: &str, begin: usize, end: usize) -> ! {
    const MAX_DISPLAY_LENGTH: usize = 256;
    let trunc_len = s.floor_char_boundary(MAX_DISPLAY_LENGTH);
    let s_trunc = &s[..trunc_len];
    let ellipsis = if trunc_len < s.len() { "[...]" } else { "" };

    // 1. out of bounds
    if begin > s.len() || end > s.len() {
        let oob_index = if begin > s.len() { begin } else { end };
        panic!("byte index {oob_index} is out of bounds of `{s_trunc}`{ellipsis}");
    }

    // 2. begin <= end
    assert!(
        begin <= end,
        "begin <= end ({} <= {}) when slicing `{}`{}",
        begin,
        end,
        s_trunc,
        ellipsis
    );

    // 3. character boundary
    let index = if !s.is_char_boundary(begin) { begin } else { end };
    // find the character
    let char_start = s.floor_char_boundary(index);
    // `char_start` must be less than len and a char boundary
    let ch = s[char_start..].chars().next().unwrap();
    let char_range = char_start..char_start + ch.len_utf8();
    panic!(
        "byte index {} is not a char boundary; it is inside {:?} (bytes {:?}) of `{}`{}",
        index, ch, char_range, s_trunc, ellipsis
    );
}

impl str {
    /// Returns the length of `self`.
    ///
    /// This length is in bytes, not [`char`]s or graphemes. In other words,
    /// it might not be what a human considers the length of the string.
    ///
    /// [`char`]: prim@char
    ///
    /// # Examples
    ///
    /// ```
    /// let len = "foo".len();
    /// assert_eq!(3, len);
    ///
    /// assert_eq!("ƒoo".len(), 4); // fancy f!
    /// assert_eq!("ƒoo".chars().count(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_str_len", since = "1.39.0")]
    #[rustc_diagnostic_item = "str_len"]
    #[rustc_no_implicit_autorefs]
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Returns `true` if `self` has a length of zero bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "";
    /// assert!(s.is_empty());
    ///
    /// let s = "not empty";
    /// assert!(!s.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_str_is_empty", since = "1.39.0")]
    #[rustc_no_implicit_autorefs]
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Converts a slice of bytes to a string slice.
    ///
    /// A string slice ([`&str`]) is made of bytes ([`u8`]), and a byte slice
    /// ([`&[u8]`][byteslice]) is made of bytes, so this function converts between
    /// the two. Not all byte slices are valid string slices, however: [`&str`] requires
    /// that it is valid UTF-8. `from_utf8()` checks to ensure that the bytes are valid
    /// UTF-8, and then does the conversion.
    ///
    /// [`&str`]: str
    /// [byteslice]: prim@slice
    ///
    /// If you are sure that the byte slice is valid UTF-8, and you don't want to
    /// incur the overhead of the validity check, there is an unsafe version of
    /// this function, [`from_utf8_unchecked`], which has the same
    /// behavior but skips the check.
    ///
    /// If you need a `String` instead of a `&str`, consider
    /// [`String::from_utf8`][string].
    ///
    /// [string]: ../std/string/struct.String.html#method.from_utf8
    ///
    /// Because you can stack-allocate a `[u8; N]`, and you can take a
    /// [`&[u8]`][byteslice] of it, this function is one way to have a
    /// stack-allocated string. There is an example of this in the
    /// examples section below.
    ///
    /// [byteslice]: slice
    ///
    /// # Errors
    ///
    /// Returns `Err` if the slice is not UTF-8 with a description as to why the
    /// provided slice is not UTF-8.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// // We can use the ? (try) operator to check if the bytes are valid
    /// let sparkle_heart = str::from_utf8(&sparkle_heart)?;
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// # Ok::<_, std::str::Utf8Error>(())
    /// ```
    ///
    /// Incorrect bytes:
    ///
    /// ```
    /// // some invalid bytes, in a vector
    /// let sparkle_heart = vec![0, 159, 146, 150];
    ///
    /// assert!(str::from_utf8(&sparkle_heart).is_err());
    /// ```
    ///
    /// See the docs for [`Utf8Error`] for more details on the kinds of
    /// errors that can be returned.
    ///
    /// A "stack allocated string":
    ///
    /// ```
    /// // some bytes, in a stack-allocated array
    /// let sparkle_heart = [240, 159, 146, 150];
    ///
    /// // We know these bytes are valid, so just use `unwrap()`.
    /// let sparkle_heart: &str = str::from_utf8(&sparkle_heart).unwrap();
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    #[stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_const_stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_diagnostic_item = "str_inherent_from_utf8"]
    pub const fn from_utf8(v: &[u8]) -> Result<&str, Utf8Error> {
        converts::from_utf8(v)
    }

    /// Converts a mutable slice of bytes to a mutable string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // "Hello, Rust!" as a mutable vector
    /// let mut hellorust = vec![72, 101, 108, 108, 111, 44, 32, 82, 117, 115, 116, 33];
    ///
    /// // As we know these bytes are valid, we can use `unwrap()`
    /// let outstr = str::from_utf8_mut(&mut hellorust).unwrap();
    ///
    /// assert_eq!("Hello, Rust!", outstr);
    /// ```
    ///
    /// Incorrect bytes:
    ///
    /// ```
    /// // Some invalid bytes in a mutable vector
    /// let mut invalid = vec![128, 223];
    ///
    /// assert!(str::from_utf8_mut(&mut invalid).is_err());
    /// ```
    /// See the docs for [`Utf8Error`] for more details on the kinds of
    /// errors that can be returned.
    #[stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_const_stable(feature = "const_str_from_utf8", since = "1.87.0")]
    #[rustc_diagnostic_item = "str_inherent_from_utf8_mut"]
    pub const fn from_utf8_mut(v: &mut [u8]) -> Result<&mut str, Utf8Error> {
        converts::from_utf8_mut(v)
    }

    /// Converts a slice of bytes to a string slice without checking
    /// that the string contains valid UTF-8.
    ///
    /// See the safe version, [`from_utf8`], for more information.
    ///
    /// # Safety
    ///
    /// The bytes passed in must be valid UTF-8.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// let sparkle_heart = unsafe {
    ///     str::from_utf8_unchecked(&sparkle_heart)
    /// };
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_const_stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_diagnostic_item = "str_inherent_from_utf8_unchecked"]
    pub const unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
        // SAFETY: converts::from_utf8_unchecked has the same safety requirements as this function.
        unsafe { converts::from_utf8_unchecked(v) }
    }

    /// Converts a slice of bytes to a string slice without checking
    /// that the string contains valid UTF-8; mutable version.
    ///
    /// See the immutable version, [`from_utf8_unchecked()`] for documentation and safety requirements.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut heart = vec![240, 159, 146, 150];
    /// let heart = unsafe { str::from_utf8_unchecked_mut(&mut heart) };
    ///
    /// assert_eq!("💖", heart);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_const_stable(feature = "inherent_str_constructors", since = "1.87.0")]
    #[rustc_diagnostic_item = "str_inherent_from_utf8_unchecked_mut"]
    pub const unsafe fn from_utf8_unchecked_mut(v: &mut [u8]) -> &mut str {
        // SAFETY: converts::from_utf8_unchecked_mut has the same safety requirements as this function.
        unsafe { converts::from_utf8_unchecked_mut(v) }
    }

    /// Checks that `index`-th byte is the first byte in a UTF-8 code point
    /// sequence or the end of the string.
    ///
    /// The start and end of the string (when `index == self.len()`) are
    /// considered to be boundaries.
    ///
    /// Returns `false` if `index` is greater than `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// assert!(s.is_char_boundary(0));
    /// // start of `老`
    /// assert!(s.is_char_boundary(6));
    /// assert!(s.is_char_boundary(s.len()));
    ///
    /// // second byte of `ö`
    /// assert!(!s.is_char_boundary(2));
    ///
    /// // third byte of `老`
    /// assert!(!s.is_char_boundary(8));
    /// ```
    #[must_use]
    #[stable(feature = "is_char_boundary", since = "1.9.0")]
    #[rustc_const_stable(feature = "const_is_char_boundary", since = "1.86.0")]
    #[inline]
    pub const fn is_char_boundary(&self, index: usize) -> bool {
        // 0 is always ok.
        // Test for 0 explicitly so that it can optimize out the check
        // easily and skip reading string data for that case.
        // Note that optimizing `self.get(..index)` relies on this.
        if index == 0 {
            return true;
        }

        if index >= self.len() {
            // For `true` we have two options:
            //
            // - index == self.len()
            //   Empty strings are valid, so return true
            // - index > self.len()
            //   In this case return false
            //
            // The check is placed exactly here, because it improves generated
            // code on higher opt-levels. See PR #84751 for more details.
            index == self.len()
        } else {
            self.as_bytes()[index].is_utf8_char_boundary()
        }
    }

    /// Finds the closest `x` not exceeding `index` where [`is_char_boundary(x)`] is `true`.
    ///
    /// This method can help you truncate a string so that it's still valid UTF-8, but doesn't
    /// exceed a given number of bytes. Note that this is done purely at the character level
    /// and can still visually split graphemes, even though the underlying characters aren't
    /// split. For example, the emoji 🧑‍🔬 (scientist) could be split so that the string only
    /// includes 🧑 (person) instead.
    ///
    /// [`is_char_boundary(x)`]: Self::is_char_boundary
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(round_char_boundary)]
    /// let s = "❤️🧡💛💚💙💜";
    /// assert_eq!(s.len(), 26);
    /// assert!(!s.is_char_boundary(13));
    ///
    /// let closest = s.floor_char_boundary(13);
    /// assert_eq!(closest, 10);
    /// assert_eq!(&s[..closest], "❤️🧡");
    /// ```
    #[unstable(feature = "round_char_boundary", issue = "93743")]
    #[inline]
    pub fn floor_char_boundary(&self, index: usize) -> usize {
        if index >= self.len() {
            self.len()
        } else {
            let lower_bound = index.saturating_sub(3);
            let new_index = self.as_bytes()[lower_bound..=index]
                .iter()
                .rposition(|b| b.is_utf8_char_boundary());

            // SAFETY: we know that the character boundary will be within four bytes
            unsafe { lower_bound + new_index.unwrap_unchecked() }
        }
    }

    /// Finds the closest `x` not below `index` where [`is_char_boundary(x)`] is `true`.
    ///
    /// If `index` is greater than the length of the string, this returns the length of the string.
    ///
    /// This method is the natural complement to [`floor_char_boundary`]. See that method
    /// for more details.
    ///
    /// [`floor_char_boundary`]: str::floor_char_boundary
    /// [`is_char_boundary(x)`]: Self::is_char_boundary
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(round_char_boundary)]
    /// let s = "❤️🧡💛💚💙💜";
    /// assert_eq!(s.len(), 26);
    /// assert!(!s.is_char_boundary(13));
    ///
    /// let closest = s.ceil_char_boundary(13);
    /// assert_eq!(closest, 14);
    /// assert_eq!(&s[..closest], "❤️🧡💛");
    /// ```
    #[unstable(feature = "round_char_boundary", issue = "93743")]
    #[inline]
    pub fn ceil_char_boundary(&self, index: usize) -> usize {
        if index > self.len() {
            self.len()
        } else {
            let upper_bound = Ord::min(index + 4, self.len());
            self.as_bytes()[index..upper_bound]
                .iter()
                .position(|b| b.is_utf8_char_boundary())
                .map_or(upper_bound, |pos| pos + index)
        }
    }

    /// Converts a string slice to a byte slice. To convert the byte slice back
    /// into a string slice, use the [`from_utf8`] function.
    ///
    /// # Examples
    ///
    /// ```
    /// let bytes = "bors".as_bytes();
    /// assert_eq!(b"bors", bytes);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "str_as_bytes", since = "1.39.0")]
    #[must_use]
    #[inline(always)]
    #[allow(unused_attributes)]
    pub const fn as_bytes(&self) -> &[u8] {
        // SAFETY: const sound because we transmute two types with the same layout
        unsafe { mem::transmute(self) }
    }

    /// Converts a mutable string slice to a mutable byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the content of the slice is valid UTF-8
    /// before the borrow ends and the underlying `str` is used.
    ///
    /// Use of a `str` whose contents are not valid UTF-8 is undefined behavior.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = String::from("Hello");
    /// let bytes = unsafe { s.as_bytes_mut() };
    ///
    /// assert_eq!(b"Hello", bytes);
    /// ```
    ///
    /// Mutability:
    ///
    /// ```
    /// let mut s = String::from("🗻∈🌏");
    ///
    /// unsafe {
    ///     let bytes = s.as_bytes_mut();
    ///
    ///     bytes[0] = 0xF0;
    ///     bytes[1] = 0x9F;
    ///     bytes[2] = 0x8D;
    ///     bytes[3] = 0x94;
    /// }
    ///
    /// assert_eq!("🍔∈🌏", s);
    /// ```
    #[stable(feature = "str_mut_extras", since = "1.20.0")]
    #[rustc_const_stable(feature = "const_str_as_mut", since = "1.83.0")]
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the cast from `&str` to `&[u8]` is safe since `str`
        // has the same layout as `&[u8]` (only std can make this guarantee).
        // The pointer dereference is safe since it comes from a mutable reference which
        // is guaranteed to be valid for writes.
        unsafe { &mut *(self as *mut str as *mut [u8]) }
    }

    /// Converts a string slice to a raw pointer.
    ///
    /// As string slices are a slice of bytes, the raw pointer points to a
    /// [`u8`]. This pointer will be pointing to the first byte of the string
    /// slice.
    ///
    /// The caller must ensure that the returned pointer is never written to.
    /// If you need to mutate the contents of the string slice, use [`as_mut_ptr`].
    ///
    /// [`as_mut_ptr`]: str::as_mut_ptr
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Hello";
    /// let ptr = s.as_ptr();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "rustc_str_as_ptr", since = "1.32.0")]
    #[rustc_never_returns_null_ptr]
    #[rustc_as_ptr]
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self as *const str as *const u8
    }

    /// Converts a mutable string slice to a raw pointer.
    ///
    /// As string slices are a slice of bytes, the raw pointer points to a
    /// [`u8`]. This pointer will be pointing to the first byte of the string
    /// slice.
    ///
    /// It is your responsibility to make sure that the string slice only gets
    /// modified in a way that it remains valid UTF-8.
    #[stable(feature = "str_as_mut_ptr", since = "1.36.0")]
    #[rustc_const_stable(feature = "const_str_as_mut", since = "1.83.0")]
    #[rustc_never_returns_null_ptr]
    #[rustc_as_ptr]
    #[must_use]
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self as *mut str as *mut u8
    }

    /// Returns a subslice of `str`.
    ///
    /// This is the non-panicking alternative to indexing the `str`. Returns
    /// [`None`] whenever equivalent indexing operation would panic.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = String::from("🗻∈🌏");
    ///
    /// assert_eq!(Some("🗻"), v.get(0..4));
    ///
    /// // indices not on UTF-8 sequence boundaries
    /// assert!(v.get(1..).is_none());
    /// assert!(v.get(..8).is_none());
    ///
    /// // out of bounds
    /// assert!(v.get(..42).is_none());
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub fn get<I: SliceIndex<str>>(&self, i: I) -> Option<&I::Output> {
        i.get(self)
    }

    /// Returns a mutable subslice of `str`.
    ///
    /// This is the non-panicking alternative to indexing the `str`. Returns
    /// [`None`] whenever equivalent indexing operation would panic.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::from("hello");
    /// // correct length
    /// assert!(v.get_mut(0..5).is_some());
    /// // out of bounds
    /// assert!(v.get_mut(..42).is_none());
    /// assert_eq!(Some("he"), v.get_mut(0..2).map(|v| &*v));
    ///
    /// assert_eq!("hello", v);
    /// {
    ///     let s = v.get_mut(0..2);
    ///     let s = s.map(|s| {
    ///         s.make_ascii_uppercase();
    ///         &*s
    ///     });
    ///     assert_eq!(Some("HE"), s);
    /// }
    /// assert_eq!("HEllo", v);
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub fn get_mut<I: SliceIndex<str>>(&mut self, i: I) -> Option<&mut I::Output> {
        i.get_mut(self)
    }

    /// Returns an unchecked subslice of `str`.
    ///
    /// This is the unchecked alternative to indexing the `str`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The starting index must not exceed the ending index;
    /// * Indexes must be within bounds of the original slice;
    /// * Indexes must lie on UTF-8 sequence boundaries.
    ///
    /// Failing that, the returned string slice may reference invalid memory or
    /// violate the invariants communicated by the `str` type.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = "🗻∈🌏";
    /// unsafe {
    ///     assert_eq!("🗻", v.get_unchecked(0..4));
    ///     assert_eq!("∈", v.get_unchecked(4..7));
    ///     assert_eq!("🌏", v.get_unchecked(7..11));
    /// }
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<str>>(&self, i: I) -> &I::Output {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked`;
        // the slice is dereferenceable because `self` is a safe reference.
        // The returned pointer is safe because impls of `SliceIndex` have to guarantee that it is.
        unsafe { &*i.get_unchecked(self) }
    }

    /// Returns a mutable, unchecked subslice of `str`.
    ///
    /// This is the unchecked alternative to indexing the `str`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The starting index must not exceed the ending index;
    /// * Indexes must be within bounds of the original slice;
    /// * Indexes must lie on UTF-8 sequence boundaries.
    ///
    /// Failing that, the returned string slice may reference invalid memory or
    /// violate the invariants communicated by the `str` type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::from("🗻∈🌏");
    /// unsafe {
    ///     assert_eq!("🗻", v.get_unchecked_mut(0..4));
    ///     assert_eq!("∈", v.get_unchecked_mut(4..7));
    ///     assert_eq!("🌏", v.get_unchecked_mut(7..11));
    /// }
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub unsafe fn get_unchecked_mut<I: SliceIndex<str>>(&mut self, i: I) -> &mut I::Output {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked_mut`;
        // the slice is dereferenceable because `self` is a safe reference.
        // The returned pointer is safe because impls of `SliceIndex` have to guarantee that it is.
        unsafe { &mut *i.get_unchecked_mut(self) }
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`str`] and [`Index`].
    ///
    /// [`Index`]: crate::ops::Index
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get a mutable string slice instead, see the
    /// [`slice_mut_unchecked`] method.
    ///
    /// [`slice_mut_unchecked`]: str::slice_mut_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisfied:
    ///
    /// * `begin` must not exceed `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// unsafe {
    ///     assert_eq!("Löwe 老虎 Léopard", s.slice_unchecked(0, 21));
    /// }
    ///
    /// let s = "Hello, world!";
    ///
    /// unsafe {
    ///     assert_eq!("world", s.slice_unchecked(7, 12));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.29.0", note = "use `get_unchecked(begin..end)` instead")]
    #[must_use]
    #[inline]
    pub unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked`;
        // the slice is dereferenceable because `self` is a safe reference.
        // The returned pointer is safe because impls of `SliceIndex` have to guarantee that it is.
        unsafe { &*(begin..end).get_unchecked(self) }
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`str`] and [`IndexMut`].
    ///
    /// [`IndexMut`]: crate::ops::IndexMut
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get an immutable string slice instead, see the
    /// [`slice_unchecked`] method.
    ///
    /// [`slice_unchecked`]: str::slice_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisfied:
    ///
    /// * `begin` must not exceed `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    #[stable(feature = "str_slice_mut", since = "1.5.0")]
    #[deprecated(since = "1.29.0", note = "use `get_unchecked_mut(begin..end)` instead")]
    #[inline]
    pub unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked_mut`;
        // the slice is dereferenceable because `self` is a safe reference.
        // The returned pointer is safe because impls of `SliceIndex` have to guarantee that it is.
        unsafe { &mut *(begin..end).get_unchecked_mut(self) }
    }

    /// Divides one string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get mutable string slices instead, see the [`split_at_mut`]
    /// method.
    ///
    /// [`split_at_mut`]: str::split_at_mut
    ///
    /// # Panics
    ///
    /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is past
    /// the end of the last code point of the string slice.  For a non-panicking
    /// alternative see [`split_at_checked`](str::split_at_checked).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Per Martin-Löf";
    ///
    /// let (first, last) = s.split_at(3);
    ///
    /// assert_eq!("Per", first);
    /// assert_eq!(" Martin-Löf", last);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    #[rustc_const_stable(feature = "const_str_split_at", since = "1.86.0")]
    pub const fn split_at(&self, mid: usize) -> (&str, &str) {
        match self.split_at_checked(mid) {
            None => slice_error_fail(self, 0, mid),
            Some(pair) => pair,
        }
    }

    /// Divides one mutable string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get immutable string slices instead, see the [`split_at`] method.
    ///
    /// [`split_at`]: str::split_at
    ///
    /// # Panics
    ///
    /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is past
    /// the end of the last code point of the string slice.  For a non-panicking
    /// alternative see [`split_at_mut_checked`](str::split_at_mut_checked).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = "Per Martin-Löf".to_string();
    /// {
    ///     let (first, last) = s.split_at_mut(3);
    ///     first.make_ascii_uppercase();
    ///     assert_eq!("PER", first);
    ///     assert_eq!(" Martin-Löf", last);
    /// }
    /// assert_eq!("PER Martin-Löf", s);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    #[rustc_const_stable(feature = "const_str_split_at", since = "1.86.0")]
    pub const fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            // SAFETY: just checked that `mid` is on a char boundary.
            unsafe { self.split_at_mut_unchecked(mid) }
        } else {
            slice_error_fail(self, 0, mid)
        }
    }

    /// Divides one string slice into two at an index.
    ///
    /// The argument, `mid`, should be a valid byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point. The
    /// method returns `None` if that’s not the case.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get mutable string slices instead, see the [`split_at_mut_checked`]
    /// method.
    ///
    /// [`split_at_mut_checked`]: str::split_at_mut_checked
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Per Martin-Löf";
    ///
    /// let (first, last) = s.split_at_checked(3).unwrap();
    /// assert_eq!("Per", first);
    /// assert_eq!(" Martin-Löf", last);
    ///
    /// assert_eq!(None, s.split_at_checked(13));  // Inside “ö”
    /// assert_eq!(None, s.split_at_checked(16));  // Beyond the string length
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "split_at_checked", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_str_split_at", since = "1.86.0")]
    pub const fn split_at_checked(&self, mid: usize) -> Option<(&str, &str)> {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            // SAFETY: just checked that `mid` is on a char boundary.
            Some(unsafe { self.split_at_unchecked(mid) })
        } else {
            None
        }
    }

    /// Divides one mutable string slice into two at an index.
    ///
    /// The argument, `mid`, should be a valid byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point. The
    /// method returns `None` if that’s not the case.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get immutable string slices instead, see the [`split_at_checked`] method.
    ///
    /// [`split_at_checked`]: str::split_at_checked
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = "Per Martin-Löf".to_string();
    /// if let Some((first, last)) = s.split_at_mut_checked(3) {
    ///     first.make_ascii_uppercase();
    ///     assert_eq!("PER", first);
    ///     assert_eq!(" Martin-Löf", last);
    /// }
    /// assert_eq!("PER Martin-Löf", s);
    ///
    /// assert_eq!(None, s.split_at_mut_checked(13));  // Inside “ö”
    /// assert_eq!(None, s.split_at_mut_checked(16));  // Beyond the string length
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "split_at_checked", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_str_split_at", since = "1.86.0")]
    pub const fn split_at_mut_checked(&mut self, mid: usize) -> Option<(&mut str, &mut str)> {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            // SAFETY: just checked that `mid` is on a char boundary.
            Some(unsafe { self.split_at_mut_unchecked(mid) })
        } else {
            None
        }
    }

    /// Divides one string slice into two at an index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `mid` is a valid byte offset from the start
    /// of the string and falls on the boundary of a UTF-8 code point.
    const unsafe fn split_at_unchecked(&self, mid: usize) -> (&str, &str) {
        let len = self.len();
        let ptr = self.as_ptr();
        // SAFETY: caller guarantees `mid` is on a char boundary.
        unsafe {
            (
                from_utf8_unchecked(slice::from_raw_parts(ptr, mid)),
                from_utf8_unchecked(slice::from_raw_parts(ptr.add(mid), len - mid)),
            )
        }
    }

    /// Divides one string slice into two at an index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `mid` is a valid byte offset from the start
    /// of the string and falls on the boundary of a UTF-8 code point.
    const unsafe fn split_at_mut_unchecked(&mut self, mid: usize) -> (&mut str, &mut str) {
        let len = self.len();
        let ptr = self.as_mut_ptr();
        // SAFETY: caller guarantees `mid` is on a char boundary.
        unsafe {
            (
                from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr, mid)),
                from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr.add(mid), len - mid)),
            )
        }
    }

    /// Returns an iterator over the [`char`]s of a string slice.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns such an iterator.
    ///
    /// It's important to remember that [`char`] represents a Unicode Scalar
    /// Value, and might not match your idea of what a 'character' is. Iteration
    /// over grapheme clusters may be what you actually want. This functionality
    /// is not provided by Rust's standard library, check crates.io instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let word = "goodbye";
    ///
    /// let count = word.chars().count();
    /// assert_eq!(7, count);
    ///
    /// let mut chars = word.chars();
    ///
    /// assert_eq!(Some('g'), chars.next());
    /// assert_eq!(Some('o'), chars.next());
    /// assert_eq!(Some('o'), chars.next());
    /// assert_eq!(Some('d'), chars.next());
    /// assert_eq!(Some('b'), chars.next());
    /// assert_eq!(Some('y'), chars.next());
    /// assert_eq!(Some('e'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    ///
    /// Remember, [`char`]s might not match your intuition about characters:
    ///
    /// [`char`]: prim@char
    ///
    /// ```
    /// let y = "y̆";
    ///
    /// let mut chars = y.chars();
    ///
    /// assert_eq!(Some('y'), chars.next()); // not 'y̆'
    /// assert_eq!(Some('\u{0306}'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[rustc_diagnostic_item = "str_chars"]
    pub fn chars(&self) -> Chars<'_> {
        Chars { iter: self.as_bytes().iter() }
    }

    /// Returns an iterator over the [`char`]s of a string slice, and their
    /// positions.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns an iterator of both
    /// these [`char`]s, as well as their byte positions.
    ///
    /// The iterator yields tuples. The position is first, the [`char`] is
    /// second.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let word = "goodbye";
    ///
    /// let count = word.char_indices().count();
    /// assert_eq!(7, count);
    ///
    /// let mut char_indices = word.char_indices();
    ///
    /// assert_eq!(Some((0, 'g')), char_indices.next());
    /// assert_eq!(Some((1, 'o')), char_indices.next());
    /// assert_eq!(Some((2, 'o')), char_indices.next());
    /// assert_eq!(Some((3, 'd')), char_indices.next());
    /// assert_eq!(Some((4, 'b')), char_indices.next());
    /// assert_eq!(Some((5, 'y')), char_indices.next());
    /// assert_eq!(Some((6, 'e')), char_indices.next());
    ///
    /// assert_eq!(None, char_indices.next());
    /// ```
    ///
    /// Remember, [`char`]s might not match your intuition about characters:
    ///
    /// [`char`]: prim@char
    ///
    /// ```
    /// let yes = "y̆es";
    ///
    /// let mut char_indices = yes.char_indices();
    ///
    /// assert_eq!(Some((0, 'y')), char_indices.next()); // not (0, 'y̆')
    /// assert_eq!(Some((1, '\u{0306}')), char_indices.next());
    ///
    /// // note the 3 here - the previous character took up two bytes
    /// assert_eq!(Some((3, 'e')), char_indices.next());
    /// assert_eq!(Some((4, 's')), char_indices.next());
    ///
    /// assert_eq!(None, char_indices.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn char_indices(&self) -> CharIndices<'_> {
        CharIndices { front_offset: 0, iter: self.chars() }
    }

    /// Returns an iterator over the bytes of a string slice.
    ///
    /// As a string slice consists of a sequence of bytes, we can iterate
    /// through a string slice by byte. This method returns such an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut bytes = "bors".bytes();
    ///
    /// assert_eq!(Some(b'b'), bytes.next());
    /// assert_eq!(Some(b'o'), bytes.next());
    /// assert_eq!(Some(b'r'), bytes.next());
    /// assert_eq!(Some(b's'), bytes.next());
    ///
    /// assert_eq!(None, bytes.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn bytes(&self) -> Bytes<'_> {
        Bytes(self.as_bytes().iter().copied())
    }

    /// Splits a string slice by whitespace.
    ///
    /// The iterator returned will return string slices that are sub-slices of
    /// the original string slice, separated by any amount of whitespace.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`. If you only want to split on ASCII whitespace
    /// instead, use [`split_ascii_whitespace`].
    ///
    /// [`split_ascii_whitespace`]: str::split_ascii_whitespace
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut iter = "A few words".split_whitespace();
    ///
    /// assert_eq!(Some("A"), iter.next());
    /// assert_eq!(Some("few"), iter.next());
    /// assert_eq!(Some("words"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// All kinds of whitespace are considered:
    ///
    /// ```
    /// let mut iter = " Mary   had\ta\u{2009}little  \n\t lamb".split_whitespace();
    /// assert_eq!(Some("Mary"), iter.next());
    /// assert_eq!(Some("had"), iter.next());
    /// assert_eq!(Some("a"), iter.next());
    /// assert_eq!(Some("little"), iter.next());
    /// assert_eq!(Some("lamb"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// If the string is empty or all whitespace, the iterator yields no string slices:
    /// ```
    /// assert_eq!("".split_whitespace().next(), None);
    /// assert_eq!("   ".split_whitespace().next(), None);
    /// ```
    #[must_use = "this returns the split string as an iterator, \
                  without modifying the original"]
    #[stable(feature = "split_whitespace", since = "1.1.0")]
    #[rustc_diagnostic_item = "str_split_whitespace"]
    #[inline]
    pub fn split_whitespace(&self) -> SplitWhitespace<'_> {
        SplitWhitespace { inner: self.split(IsWhitespace).filter(IsNotEmpty) }
    }

    /// Splits a string slice by ASCII whitespace.
    ///
    /// The iterator returned will return string slices that are sub-slices of
    /// the original string slice, separated by any amount of ASCII whitespace.
    ///
    /// To split by Unicode `Whitespace` instead, use [`split_whitespace`].
    ///
    /// [`split_whitespace`]: str::split_whitespace
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut iter = "A few words".split_ascii_whitespace();
    ///
    /// assert_eq!(Some("A"), iter.next());
    /// assert_eq!(Some("few"), iter.next());
    /// assert_eq!(Some("words"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// All kinds of ASCII whitespace are considered:
    ///
    /// ```
    /// let mut iter = " Mary   had\ta little  \n\t lamb".split_ascii_whitespace();
    /// assert_eq!(Some("Mary"), iter.next());
    /// assert_eq!(Some("had"), iter.next());
    /// assert_eq!(Some("a"), iter.next());
    /// assert_eq!(Some("little"), iter.next());
    /// assert_eq!(Some("lamb"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// If the string is empty or all ASCII whitespace, the iterator yields no string slices:
    /// ```
    /// assert_eq!("".split_ascii_whitespace().next(), None);
    /// assert_eq!("   ".split_ascii_whitespace().next(), None);
    /// ```
    #[must_use = "this returns the split string as an iterator, \
                  without modifying the original"]
    #[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
    #[inline]
    pub fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace<'_> {
        let inner =
            self.as_bytes().split(IsAsciiWhitespace).filter(BytesIsNotEmpty).map(UnsafeBytesToStr);
        SplitAsciiWhitespace { inner }
    }

    /// Returns an iterator over the lines of a string, as string slices.
    ///
    /// Lines are split at line endings that are either newlines (`\n`) or
    /// sequences of a carriage return followed by a line feed (`\r\n`).
    ///
    /// Line terminators are not included in the lines returned by the iterator.
    ///
    /// Note that any carriage return (`\r`) not immediately followed by a
    /// line feed (`\n`) does not split a line. These carriage returns are
    /// thereby included in the produced lines.
    ///
    /// The final line ending is optional. A string that ends with a final line
    /// ending will return the same lines as an otherwise identical string
    /// without a final line ending.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let text = "foo\r\nbar\n\nbaz\r";
    /// let mut lines = text.lines();
    ///
    /// assert_eq!(Some("foo"), lines.next());
    /// assert_eq!(Some("bar"), lines.next());
    /// assert_eq!(Some(""), lines.next());
    /// // Trailing carriage return is included in the last line
    /// assert_eq!(Some("baz\r"), lines.next());
    ///
    /// assert_eq!(None, lines.next());
    /// ```
    ///
    /// The final line does not require any ending:
    ///
    /// ```
    /// let text = "foo\nbar\n\r\nbaz";
    /// let mut lines = text.lines();
    ///
    /// assert_eq!(Some("foo"), lines.next());
    /// assert_eq!(Some("bar"), lines.next());
    /// assert_eq!(Some(""), lines.next());
    /// assert_eq!(Some("baz"), lines.next());
    ///
    /// assert_eq!(None, lines.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn lines(&self) -> Lines<'_> {
        Lines(self.split_inclusive('\n').map(LinesMap))
    }

    /// Returns an iterator over the lines of a string.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.4.0", note = "use lines() instead now", suggestion = "lines")]
    #[inline]
    #[allow(deprecated)]
    pub fn lines_any(&self) -> LinesAny<'_> {
        LinesAny(self.lines())
    }

    /// Returns an iterator of `u16` over the string encoded
    /// as native endian UTF-16 (without byte-order mark).
    ///
    /// # Examples
    ///
    /// ```
    /// let text = "Zażółć gęślą jaźń";
    ///
    /// let utf8_len = text.len();
    /// let utf16_len = text.encode_utf16().count();
    ///
    /// assert!(utf16_len <= utf8_len);
    /// ```
    #[must_use = "this returns the encoded string as an iterator, \
                  without modifying the original"]
    #[stable(feature = "encode_utf16", since = "1.8.0")]
    pub fn encode_utf16(&self) -> EncodeUtf16<'_> {
        EncodeUtf16 { chars: self.chars(), extra: 0 }
    }

    /// Returns `true` if the given pattern matches a sub-slice of
    /// this string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.contains("nana"));
    /// assert!(!bananas.contains("apples"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn contains<P: Pattern>(&self, pat: P) -> bool {
        pat.is_contained_in(self)
    }

    /// Returns `true` if the given pattern matches a prefix of this
    /// string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// The [pattern] can be a `&str`, in which case this function will return true if
    /// the `&str` is a prefix of this string slice.
    ///
    /// The [pattern] can also be a [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    /// These will only be checked against the first character of this string slice.
    /// Look at the second example below regarding behavior for slices of [`char`]s.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.starts_with("bana"));
    /// assert!(!bananas.starts_with("nana"));
    /// ```
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// // Note that both of these assert successfully.
    /// assert!(bananas.starts_with(&['b', 'a', 'n', 'a']));
    /// assert!(bananas.starts_with(&['a', 'b', 'c', 'd']));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "str_starts_with"]
    pub fn starts_with<P: Pattern>(&self, pat: P) -> bool {
        pat.is_prefix_of(self)
    }

    /// Returns `true` if the given pattern matches a suffix of this
    /// string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.ends_with("anas"));
    /// assert!(!bananas.ends_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "str_ends_with"]
    pub fn ends_with<P: Pattern>(&self, pat: P) -> bool
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        pat.is_suffix_of(self)
    }

    /// Returns the byte index of the first character of this string slice that
    /// matches the pattern.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard Gepardi";
    ///
    /// assert_eq!(s.find('L'), Some(0));
    /// assert_eq!(s.find('é'), Some(14));
    /// assert_eq!(s.find("pard"), Some(17));
    /// ```
    ///
    /// More complex patterns using point-free style and closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find(char::is_whitespace), Some(5));
    /// assert_eq!(s.find(char::is_lowercase), Some(1));
    /// assert_eq!(s.find(|c: char| c.is_whitespace() || c.is_lowercase()), Some(1));
    /// assert_eq!(s.find(|c: char| (c < 'o') && (c > 'a')), Some(4));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.find(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn find<P: Pattern>(&self, pat: P) -> Option<usize> {
        pat.into_searcher(self).next_match().map(|(i, _)| i)
    }

    /// Returns the byte index for the first character of the last match of the pattern in
    /// this string slice.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard Gepardi";
    ///
    /// assert_eq!(s.rfind('L'), Some(13));
    /// assert_eq!(s.rfind('é'), Some(14));
    /// assert_eq!(s.rfind("pard"), Some(24));
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind(char::is_whitespace), Some(12));
    /// assert_eq!(s.rfind(char::is_lowercase), Some(20));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.rfind(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rfind<P: Pattern>(&self, pat: P) -> Option<usize>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        pat.into_searcher(self).next_match_back().map(|(i, _)| i)
    }

    /// Returns an iterator over substrings of this string slice, separated by
    /// characters matched by a pattern.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, e.g., [`char`], but not for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rsplit`] method can be used.
    ///
    /// [`rsplit`]: str::rsplit
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".split(' ').collect();
    /// assert_eq!(v, ["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: Vec<&str> = "".split('X').collect();
    /// assert_eq!(v, [""]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').collect();
    /// assert_eq!(v, ["lion", "", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".split("::").collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".split(char::is_numeric).collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    ///
    /// let v: Vec<&str> = "lionXtigerXleopard".split(char::is_uppercase).collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    /// ```
    ///
    /// If the pattern is a slice of chars, split on each occurrence of any of the characters:
    ///
    /// ```
    /// let v: Vec<&str> = "2020-11-03 23:59".split(&['-', ' ', ':', '@'][..]).collect();
    /// assert_eq!(v, ["2020", "11", "03", "23", "59"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".split(|c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    /// ```
    ///
    /// If a string contains multiple contiguous separators, you will end up
    /// with empty strings in the output:
    ///
    /// ```
    /// let x = "||||a||b|c".to_string();
    /// let d: Vec<_> = x.split('|').collect();
    ///
    /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    /// ```
    ///
    /// Contiguous separators are separated by the empty string.
    ///
    /// ```
    /// let x = "(///)".to_string();
    /// let d: Vec<_> = x.split('/').collect();
    ///
    /// assert_eq!(d, &["(", "", "", ")"]);
    /// ```
    ///
    /// Separators at the start or end of a string are neighbored
    /// by empty strings.
    ///
    /// ```
    /// let d: Vec<_> = "010".split("0").collect();
    /// assert_eq!(d, &["", "1", ""]);
    /// ```
    ///
    /// When the empty string is used as a separator, it separates
    /// every character in the string, along with the beginning
    /// and end of the string.
    ///
    /// ```
    /// let f: Vec<_> = "rust".split("").collect();
    /// assert_eq!(f, &["", "r", "u", "s", "t", ""]);
    /// ```
    ///
    /// Contiguous separators can lead to possibly surprising behavior
    /// when whitespace is used as the separator. This code is correct:
    ///
    /// ```
    /// let x = "    a  b c".to_string();
    /// let d: Vec<_> = x.split(' ').collect();
    ///
    /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    /// ```
    ///
    /// It does _not_ give you:
    ///
    /// ```,ignore
    /// assert_eq!(d, &["a", "b", "c"]);
    /// ```
    ///
    /// Use [`split_whitespace`] for this behavior.
    ///
    /// [`split_whitespace`]: str::split_whitespace
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split<P: Pattern>(&self, pat: P) -> Split<'_, P> {
        Split(SplitInternal {
            start: 0,
            end: self.len(),
            matcher: pat.into_searcher(self),
            allow_trailing_empty: true,
            finished: false,
        })
    }

    /// Returns an iterator over substrings of this string slice, separated by
    /// characters matched by a pattern.
    ///
    /// Differs from the iterator produced by `split` in that `split_inclusive`
    /// leaves the matched part as the terminator of the substring.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb\nlittle lamb\nlittle lamb."
    ///     .split_inclusive('\n').collect();
    /// assert_eq!(v, ["Mary had a little lamb\n", "little lamb\n", "little lamb."]);
    /// ```
    ///
    /// If the last element of the string is matched,
    /// that element will be considered the terminator of the preceding substring.
    /// That substring will be the last item returned by the iterator.
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb\nlittle lamb\nlittle lamb.\n"
    ///     .split_inclusive('\n').collect();
    /// assert_eq!(v, ["Mary had a little lamb\n", "little lamb\n", "little lamb.\n"]);
    /// ```
    #[stable(feature = "split_inclusive", since = "1.51.0")]
    #[inline]
    pub fn split_inclusive<P: Pattern>(&self, pat: P) -> SplitInclusive<'_, P> {
        SplitInclusive(SplitInternal {
            start: 0,
            end: self.len(),
            matcher: pat.into_searcher(self),
            allow_trailing_empty: false,
            finished: false,
        })
    }

    /// Returns an iterator over substrings of the given string slice, separated
    /// by characters matched by a pattern and yielded in reverse order.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// For iterating from the front, the [`split`] method can be used.
    ///
    /// [`split`]: str::split
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".rsplit(' ').collect();
    /// assert_eq!(v, ["lamb", "little", "a", "had", "Mary"]);
    ///
    /// let v: Vec<&str> = "".rsplit('X').collect();
    /// assert_eq!(v, [""]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplit('X').collect();
    /// assert_eq!(v, ["leopard", "tiger", "", "lion"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".rsplit("::").collect();
    /// assert_eq!(v, ["leopard", "tiger", "lion"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".rsplit(|c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["ghi", "def", "abc"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplit<P: Pattern>(&self, pat: P) -> RSplit<'_, P>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        RSplit(self.split(pat).0)
    }

    /// Returns an iterator over substrings of the given string slice, separated
    /// by characters matched by a pattern.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// Equivalent to [`split`], except that the trailing substring
    /// is skipped if empty.
    ///
    /// [`split`]: str::split
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, e.g., [`char`], but not for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rsplit_terminator`] method can be used.
    ///
    /// [`rsplit_terminator`]: str::rsplit_terminator
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".split_terminator('.').collect();
    /// assert_eq!(v, ["A", "B"]);
    ///
    /// let v: Vec<&str> = "A..B..".split_terminator(".").collect();
    /// assert_eq!(v, ["A", "", "B", ""]);
    ///
    /// let v: Vec<&str> = "A.B:C.D".split_terminator(&['.', ':'][..]).collect();
    /// assert_eq!(v, ["A", "B", "C", "D"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_terminator<P: Pattern>(&self, pat: P) -> SplitTerminator<'_, P> {
        SplitTerminator(SplitInternal { allow_trailing_empty: false, ..self.split(pat).0 })
    }

    /// Returns an iterator over substrings of `self`, separated by characters
    /// matched by a pattern and yielded in reverse order.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// Equivalent to [`split`], except that the trailing substring is
    /// skipped if empty.
    ///
    /// [`split`]: str::split
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a
    /// reverse search, and it will be double ended if a forward/reverse
    /// search yields the same elements.
    ///
    /// For iterating from the front, the [`split_terminator`] method can be
    /// used.
    ///
    /// [`split_terminator`]: str::split_terminator
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".rsplit_terminator('.').collect();
    /// assert_eq!(v, ["B", "A"]);
    ///
    /// let v: Vec<&str> = "A..B..".rsplit_terminator(".").collect();
    /// assert_eq!(v, ["", "B", "", "A"]);
    ///
    /// let v: Vec<&str> = "A.B:C.D".rsplit_terminator(&['.', ':'][..]).collect();
    /// assert_eq!(v, ["D", "C", "B", "A"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplit_terminator<P: Pattern>(&self, pat: P) -> RSplitTerminator<'_, P>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        RSplitTerminator(self.split_terminator(pat).0)
    }

    /// Returns an iterator over substrings of the given string slice, separated
    /// by a pattern, restricted to returning at most `n` items.
    ///
    /// If `n` substrings are returned, the last substring (the `n`th substring)
    /// will contain the remainder of the string.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is
    /// not efficient to support.
    ///
    /// If the pattern allows a reverse search, the [`rsplitn`] method can be
    /// used.
    ///
    /// [`rsplitn`]: str::rsplitn
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lambda".splitn(3, ' ').collect();
    /// assert_eq!(v, ["Mary", "had", "a little lambda"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".splitn(3, "X").collect();
    /// assert_eq!(v, ["lion", "", "tigerXleopard"]);
    ///
    /// let v: Vec<&str> = "abcXdef".splitn(1, 'X').collect();
    /// assert_eq!(v, ["abcXdef"]);
    ///
    /// let v: Vec<&str> = "".splitn(1, 'X').collect();
    /// assert_eq!(v, [""]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".splitn(2, |c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["abc", "defXghi"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn<P: Pattern>(&self, n: usize, pat: P) -> SplitN<'_, P> {
        SplitN(SplitNInternal { iter: self.split(pat).0, count: n })
    }

    /// Returns an iterator over substrings of this string slice, separated by a
    /// pattern, starting from the end of the string, restricted to returning at
    /// most `n` items.
    ///
    /// If `n` substrings are returned, the last substring (the `n`th substring)
    /// will contain the remainder of the string.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is not
    /// efficient to support.
    ///
    /// For splitting from the front, the [`splitn`] method can be used.
    ///
    /// [`splitn`]: str::splitn
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".rsplitn(3, ' ').collect();
    /// assert_eq!(v, ["lamb", "little", "Mary had a"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplitn(3, 'X').collect();
    /// assert_eq!(v, ["leopard", "tiger", "lionX"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".rsplitn(2, "::").collect();
    /// assert_eq!(v, ["leopard", "lion::tiger"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".rsplitn(2, |c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["ghi", "abc1def"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn<P: Pattern>(&self, n: usize, pat: P) -> RSplitN<'_, P>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        RSplitN(self.splitn(n, pat).0)
    }

    /// Splits the string on the first occurrence of the specified delimiter and
    /// returns prefix before delimiter and suffix after delimiter.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("cfg".split_once('='), None);
    /// assert_eq!("cfg=".split_once('='), Some(("cfg", "")));
    /// assert_eq!("cfg=foo".split_once('='), Some(("cfg", "foo")));
    /// assert_eq!("cfg=foo=bar".split_once('='), Some(("cfg", "foo=bar")));
    /// ```
    #[stable(feature = "str_split_once", since = "1.52.0")]
    #[inline]
    pub fn split_once<P: Pattern>(&self, delimiter: P) -> Option<(&'_ str, &'_ str)> {
        let (start, end) = delimiter.into_searcher(self).next_match()?;
        // SAFETY: `Searcher` is known to return valid indices.
        unsafe { Some((self.get_unchecked(..start), self.get_unchecked(end..))) }
    }

    /// Splits the string on the last occurrence of the specified delimiter and
    /// returns prefix before delimiter and suffix after delimiter.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("cfg".rsplit_once('='), None);
    /// assert_eq!("cfg=foo".rsplit_once('='), Some(("cfg", "foo")));
    /// assert_eq!("cfg=foo=bar".rsplit_once('='), Some(("cfg=foo", "bar")));
    /// ```
    #[stable(feature = "str_split_once", since = "1.52.0")]
    #[inline]
    pub fn rsplit_once<P: Pattern>(&self, delimiter: P) -> Option<(&'_ str, &'_ str)>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        let (start, end) = delimiter.into_searcher(self).next_match_back()?;
        // SAFETY: `Searcher` is known to return valid indices.
        unsafe { Some((self.get_unchecked(..start), self.get_unchecked(end..))) }
    }

    /// Returns an iterator over the disjoint matches of a pattern within the
    /// given string slice.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, e.g., [`char`], but not for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatches`] method can be used.
    ///
    /// [`rmatches`]: str::rmatches
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".matches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".matches(char::is_numeric).collect();
    /// assert_eq!(v, ["1", "2", "3"]);
    /// ```
    #[stable(feature = "str_matches", since = "1.2.0")]
    #[inline]
    pub fn matches<P: Pattern>(&self, pat: P) -> Matches<'_, P> {
        Matches(MatchesInternal(pat.into_searcher(self)))
    }

    /// Returns an iterator over the disjoint matches of a pattern within this
    /// string slice, yielded in reverse order.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// For iterating from the front, the [`matches`] method can be used.
    ///
    /// [`matches`]: str::matches
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".rmatches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".rmatches(char::is_numeric).collect();
    /// assert_eq!(v, ["3", "2", "1"]);
    /// ```
    #[stable(feature = "str_matches", since = "1.2.0")]
    #[inline]
    pub fn rmatches<P: Pattern>(&self, pat: P) -> RMatches<'_, P>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        RMatches(self.matches(pat).0)
    }

    /// Returns an iterator over the disjoint matches of a pattern within this string
    /// slice as well as the index that the match starts at.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the first match are returned.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, e.g., [`char`], but not for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatch_indices`] method can be used.
    ///
    /// [`rmatch_indices`]: str::rmatch_indices
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<_> = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, [(0, "abc"), (6, "abc"), (12, "abc")]);
    ///
    /// let v: Vec<_> = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, [(1, "abc"), (4, "abc")]);
    ///
    /// let v: Vec<_> = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, [(0, "aba")]); // only the first `aba`
    /// ```
    #[stable(feature = "str_match_indices", since = "1.5.0")]
    #[inline]
    pub fn match_indices<P: Pattern>(&self, pat: P) -> MatchIndices<'_, P> {
        MatchIndices(MatchIndicesInternal(pat.into_searcher(self)))
    }

    /// Returns an iterator over the disjoint matches of a pattern within `self`,
    /// yielded in reverse order along with the index of the match.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the last match are returned.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// For iterating from the front, the [`match_indices`] method can be used.
    ///
    /// [`match_indices`]: str::match_indices
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<_> = "abcXXXabcYYYabc".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(12, "abc"), (6, "abc"), (0, "abc")]);
    ///
    /// let v: Vec<_> = "1abcabc2".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(4, "abc"), (1, "abc")]);
    ///
    /// let v: Vec<_> = "ababa".rmatch_indices("aba").collect();
    /// assert_eq!(v, [(2, "aba")]); // only the last `aba`
    /// ```
    #[stable(feature = "str_match_indices", since = "1.5.0")]
    #[inline]
    pub fn rmatch_indices<P: Pattern>(&self, pat: P) -> RMatchIndices<'_, P>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        RMatchIndices(self.match_indices(pat).0)
    }

    /// Returns a string slice with leading and trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`, which includes newlines.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "\n Hello\tworld\t\n";
    ///
    /// assert_eq!("Hello\tworld", s.trim());
    /// ```
    #[inline]
    #[must_use = "this returns the trimmed string as a slice, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "str_trim"]
    pub fn trim(&self) -> &str {
        self.trim_matches(char::is_whitespace)
    }

    /// Returns a string slice with leading whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`, which includes newlines.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. `start` in this context means the first
    /// position of that byte string; for a left-to-right language like English or
    /// Russian, this will be left side, and for right-to-left languages like
    /// Arabic or Hebrew, this will be the right side.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "\n Hello\tworld\t\n";
    /// assert_eq!("Hello\tworld\t\n", s.trim_start());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English  ";
    /// assert!(Some('E') == s.trim_start().chars().next());
    ///
    /// let s = "  עברית  ";
    /// assert!(Some('ע') == s.trim_start().chars().next());
    /// ```
    #[inline]
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    #[rustc_diagnostic_item = "str_trim_start"]
    pub fn trim_start(&self) -> &str {
        self.trim_start_matches(char::is_whitespace)
    }

    /// Returns a string slice with trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`, which includes newlines.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. `end` in this context means the last
    /// position of that byte string; for a left-to-right language like English or
    /// Russian, this will be right side, and for right-to-left languages like
    /// Arabic or Hebrew, this will be the left side.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "\n Hello\tworld\t\n";
    /// assert_eq!("\n Hello\tworld", s.trim_end());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English  ";
    /// assert!(Some('h') == s.trim_end().chars().rev().next());
    ///
    /// let s = "  עברית  ";
    /// assert!(Some('ת') == s.trim_end().chars().rev().next());
    /// ```
    #[inline]
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    #[rustc_diagnostic_item = "str_trim_end"]
    pub fn trim_end(&self) -> &str {
        self.trim_end_matches(char::is_whitespace)
    }

    /// Returns a string slice with leading whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!("Hello\tworld\t", s.trim_left());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English";
    /// assert!(Some('E') == s.trim_left().chars().next());
    ///
    /// let s = "  עברית";
    /// assert!(Some('ע') == s.trim_left().chars().next());
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.33.0", note = "superseded by `trim_start`", suggestion = "trim_start")]
    pub fn trim_left(&self) -> &str {
        self.trim_start()
    }

    /// Returns a string slice with trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!(" Hello\tworld", s.trim_right());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "English  ";
    /// assert!(Some('h') == s.trim_right().chars().rev().next());
    ///
    /// let s = "עברית  ";
    /// assert!(Some('ת') == s.trim_right().chars().rev().next());
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.33.0", note = "superseded by `trim_end`", suggestion = "trim_end")]
    pub fn trim_right(&self) -> &str {
        self.trim_end()
    }

    /// Returns a string slice with all prefixes and suffixes that match a
    /// pattern repeatedly removed.
    ///
    /// The [pattern] can be a [`char`], a slice of [`char`]s, or a function
    /// or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    /// assert_eq!("123foo1bar123".trim_matches(char::is_numeric), "foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_matches(x), "foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1foo1barXX".trim_matches(|c| c == '1' || c == 'X'), "foo1bar");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_matches<P: Pattern>(&self, pat: P) -> &str
    where
        for<'a> P::Searcher<'a>: DoubleEndedSearcher<'a>,
    {
        let mut i = 0;
        let mut j = 0;
        let mut matcher = pat.into_searcher(self);
        if let Some((a, b)) = matcher.next_reject() {
            i = a;
            j = b; // Remember earliest known match, correct it below if
            // last match is different
        }
        if let Some((_, b)) = matcher.next_reject_back() {
            j = b;
        }
        // SAFETY: `Searcher` is known to return valid indices.
        unsafe { self.get_unchecked(i..j) }
    }

    /// Returns a string slice with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. `start` in this context means the first
    /// position of that byte string; for a left-to-right language like English or
    /// Russian, this will be left side, and for right-to-left languages like
    /// Arabic or Hebrew, this will be the right side.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_start_matches('1'), "foo1bar11");
    /// assert_eq!("123foo1bar123".trim_start_matches(char::is_numeric), "foo1bar123");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_start_matches(x), "foo1bar12");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    pub fn trim_start_matches<P: Pattern>(&self, pat: P) -> &str {
        let mut i = self.len();
        let mut matcher = pat.into_searcher(self);
        if let Some((a, _)) = matcher.next_reject() {
            i = a;
        }
        // SAFETY: `Searcher` is known to return valid indices.
        unsafe { self.get_unchecked(i..self.len()) }
    }

    /// Returns a string slice with the prefix removed.
    ///
    /// If the string starts with the pattern `prefix`, returns the substring after the prefix,
    /// wrapped in `Some`. Unlike [`trim_start_matches`], this method removes the prefix exactly once.
    ///
    /// If the string does not start with `prefix`, returns `None`.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    /// [`trim_start_matches`]: Self::trim_start_matches
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("foo:bar".strip_prefix("foo:"), Some("bar"));
    /// assert_eq!("foo:bar".strip_prefix("bar"), None);
    /// assert_eq!("foofoo".strip_prefix("foo"), Some("foo"));
    /// ```
    #[must_use = "this returns the remaining substring as a new slice, \
                  without modifying the original"]
    #[stable(feature = "str_strip", since = "1.45.0")]
    pub fn strip_prefix<P: Pattern>(&self, prefix: P) -> Option<&str> {
        prefix.strip_prefix_of(self)
    }

    /// Returns a string slice with the suffix removed.
    ///
    /// If the string ends with the pattern `suffix`, returns the substring before the suffix,
    /// wrapped in `Some`.  Unlike [`trim_end_matches`], this method removes the suffix exactly once.
    ///
    /// If the string does not end with `suffix`, returns `None`.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    /// [`trim_end_matches`]: Self::trim_end_matches
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("bar:foo".strip_suffix(":foo"), Some("bar"));
    /// assert_eq!("bar:foo".strip_suffix("bar"), None);
    /// assert_eq!("foofoo".strip_suffix("foo"), Some("foo"));
    /// ```
    #[must_use = "this returns the remaining substring as a new slice, \
                  without modifying the original"]
    #[stable(feature = "str_strip", since = "1.45.0")]
    pub fn strip_suffix<P: Pattern>(&self, suffix: P) -> Option<&str>
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        suffix.strip_suffix_of(self)
    }

    /// Returns a string slice with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. `end` in this context means the last
    /// position of that byte string; for a left-to-right language like English or
    /// Russian, this will be right side, and for right-to-left languages like
    /// Arabic or Hebrew, this will be the left side.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_end_matches('1'), "11foo1bar");
    /// assert_eq!("123foo1bar123".trim_end_matches(char::is_numeric), "123foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_end_matches(x), "12foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1fooX".trim_end_matches(|c| c == '1' || c == 'X'), "1foo");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    pub fn trim_end_matches<P: Pattern>(&self, pat: P) -> &str
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        let mut j = 0;
        let mut matcher = pat.into_searcher(self);
        if let Some((_, b)) = matcher.next_reject_back() {
            j = b;
        }
        // SAFETY: `Searcher` is known to return valid indices.
        unsafe { self.get_unchecked(0..j) }
    }

    /// Returns a string slice with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
    /// assert_eq!("123foo1bar123".trim_left_matches(char::is_numeric), "foo1bar123");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_left_matches(x), "foo1bar12");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(
        since = "1.33.0",
        note = "superseded by `trim_start_matches`",
        suggestion = "trim_start_matches"
    )]
    pub fn trim_left_matches<P: Pattern>(&self, pat: P) -> &str {
        self.trim_start_matches(pat)
    }

    /// Returns a string slice with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The [pattern] can be a `&str`, [`char`], a slice of [`char`]s, or a
    /// function or closure that determines if a character matches.
    ///
    /// [`char`]: prim@char
    /// [pattern]: self::pattern
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
    /// assert_eq!("123foo1bar123".trim_right_matches(char::is_numeric), "123foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_right_matches(x), "12foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1fooX".trim_right_matches(|c| c == '1' || c == 'X'), "1foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(
        since = "1.33.0",
        note = "superseded by `trim_end_matches`",
        suggestion = "trim_end_matches"
    )]
    pub fn trim_right_matches<P: Pattern>(&self, pat: P) -> &str
    where
        for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
    {
        self.trim_end_matches(pat)
    }

    /// Parses this string slice into another type.
    ///
    /// Because `parse` is so general, it can cause problems with type
    /// inference. As such, `parse` is one of the few times you'll see
    /// the syntax affectionately known as the 'turbofish': `::<>`. This
    /// helps the inference algorithm understand specifically which type
    /// you're trying to parse into.
    ///
    /// `parse` can parse into any type that implements the [`FromStr`] trait.

    ///
    /// # Errors
    ///
    /// Will return [`Err`] if it's not possible to parse this string slice into
    /// the desired type.
    ///
    /// [`Err`]: FromStr::Err
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let four: u32 = "4".parse().unwrap();
    ///
    /// assert_eq!(4, four);
    /// ```
    ///
    /// Using the 'turbofish' instead of annotating `four`:
    ///
    /// ```
    /// let four = "4".parse::<u32>();
    ///
    /// assert_eq!(Ok(4), four);
    /// ```
    ///
    /// Failing to parse:
    ///
    /// ```
    /// let nope = "j".parse::<u32>();
    ///
    /// assert!(nope.is_err());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
        FromStr::from_str(self)
    }

    /// Checks if all characters in this string are within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = "hello!\n";
    /// let non_ascii = "Grüße, Jürgen ❤";
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_slice_is_ascii", since = "1.74.0")]
    #[must_use]
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        // We can treat each byte as character here: all multibyte characters
        // start with a byte that is not in the ASCII range, so we will stop
        // there already.
        self.as_bytes().is_ascii()
    }

    /// If this string slice [`is_ascii`](Self::is_ascii), returns it as a slice
    /// of [ASCII characters](`ascii::Char`), otherwise returns `None`.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[must_use]
    #[inline]
    pub const fn as_ascii(&self) -> Option<&[ascii::Char]> {
        // Like in `is_ascii`, we can work on the bytes directly.
        self.as_bytes().as_ascii()
    }

    /// Checks that two strings are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!("Ferris".eq_ignore_ascii_case("FERRIS"));
    /// assert!("Ferrös".eq_ignore_ascii_case("FERRöS"));
    /// assert!(!"Ferrös".eq_ignore_ascii_case("FERRÖS"));
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_unstable(feature = "const_eq_ignore_ascii_case", issue = "131719")]
    #[must_use]
    #[inline]
    pub const fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    }

    /// Converts this string to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase()`].
    ///
    /// [`to_ascii_uppercase()`]: #method.to_ascii_uppercase
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("Grüße, Jürgen ❤");
    ///
    /// s.make_ascii_uppercase();
    ///
    /// assert_eq!("GRüßE, JüRGEN ❤", s);
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_make_ascii", since = "1.84.0")]
    #[inline]
    pub const fn make_ascii_uppercase(&mut self) {
        // SAFETY: changing ASCII letters only does not invalidate UTF-8.
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_uppercase()
    }

    /// Converts this string to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase()`].
    ///
    /// [`to_ascii_lowercase()`]: #method.to_ascii_lowercase
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("GRÜßE, JÜRGEN ❤");
    ///
    /// s.make_ascii_lowercase();
    ///
    /// assert_eq!("grÜße, jÜrgen ❤", s);
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_make_ascii", since = "1.84.0")]
    #[inline]
    pub const fn make_ascii_lowercase(&mut self) {
        // SAFETY: changing ASCII letters only does not invalidate UTF-8.
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_lowercase()
    }

    /// Returns a string slice with leading ASCII whitespace removed.
    ///
    /// 'Whitespace' refers to the definition used by
    /// [`u8::is_ascii_whitespace`].
    ///
    /// [`u8::is_ascii_whitespace`]: u8::is_ascii_whitespace
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(" \t \u{3000}hello world\n".trim_ascii_start(), "\u{3000}hello world\n");
    /// assert_eq!("  ".trim_ascii_start(), "");
    /// assert_eq!("".trim_ascii_start(), "");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[rustc_const_stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[inline]
    pub const fn trim_ascii_start(&self) -> &str {
        // SAFETY: Removing ASCII characters from a `&str` does not invalidate
        // UTF-8.
        unsafe { core::str::from_utf8_unchecked(self.as_bytes().trim_ascii_start()) }
    }

    /// Returns a string slice with trailing ASCII whitespace removed.
    ///
    /// 'Whitespace' refers to the definition used by
    /// [`u8::is_ascii_whitespace`].
    ///
    /// [`u8::is_ascii_whitespace`]: u8::is_ascii_whitespace
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("\r hello world\u{3000}\n ".trim_ascii_end(), "\r hello world\u{3000}");
    /// assert_eq!("  ".trim_ascii_end(), "");
    /// assert_eq!("".trim_ascii_end(), "");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[rustc_const_stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[inline]
    pub const fn trim_ascii_end(&self) -> &str {
        // SAFETY: Removing ASCII characters from a `&str` does not invalidate
        // UTF-8.
        unsafe { core::str::from_utf8_unchecked(self.as_bytes().trim_ascii_end()) }
    }

    /// Returns a string slice with leading and trailing ASCII whitespace
    /// removed.
    ///
    /// 'Whitespace' refers to the definition used by
    /// [`u8::is_ascii_whitespace`].
    ///
    /// [`u8::is_ascii_whitespace`]: u8::is_ascii_whitespace
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("\r hello world\n ".trim_ascii(), "hello world");
    /// assert_eq!("  ".trim_ascii(), "");
    /// assert_eq!("".trim_ascii(), "");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[rustc_const_stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[inline]
    pub const fn trim_ascii(&self) -> &str {
        // SAFETY: Removing ASCII characters from a `&str` does not invalidate
        // UTF-8.
        unsafe { core::str::from_utf8_unchecked(self.as_bytes().trim_ascii()) }
    }

    /// Returns an iterator that escapes each char in `self` with [`char::escape_debug`].
    ///
    /// Note: only extended grapheme codepoints that begin the string will be
    /// escaped.
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in "❤\n!".escape_debug() {
    ///     print!("{c}");
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", "❤\n!".escape_debug());
    /// ```
    ///
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("❤\\n!");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!("❤\n!".escape_debug().to_string(), "❤\\n!");
    /// ```
    #[must_use = "this returns the escaped string as an iterator, \
                  without modifying the original"]
    #[stable(feature = "str_escape", since = "1.34.0")]
    pub fn escape_debug(&self) -> EscapeDebug<'_> {
        let mut chars = self.chars();
        EscapeDebug {
            inner: chars
                .next()
                .map(|first| first.escape_debug_ext(EscapeDebugExtArgs::ESCAPE_ALL))
                .into_iter()
                .flatten()
                .chain(chars.flat_map(CharEscapeDebugContinue)),
        }
    }

    /// Returns an iterator that escapes each char in `self` with [`char::escape_default`].
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in "❤\n!".escape_default() {
    ///     print!("{c}");
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", "❤\n!".escape_default());
    /// ```
    ///
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("\\u{{2764}}\\n!");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!("❤\n!".escape_default().to_string(), "\\u{2764}\\n!");
    /// ```
    #[must_use = "this returns the escaped string as an iterator, \
                  without modifying the original"]
    #[stable(feature = "str_escape", since = "1.34.0")]
    pub fn escape_default(&self) -> EscapeDefault<'_> {
        EscapeDefault { inner: self.chars().flat_map(CharEscapeDefault) }
    }

    /// Returns an iterator that escapes each char in `self` with [`char::escape_unicode`].
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in "❤\n!".escape_unicode() {
    ///     print!("{c}");
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", "❤\n!".escape_unicode());
    /// ```
    ///
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("\\u{{2764}}\\u{{a}}\\u{{21}}");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!("❤\n!".escape_unicode().to_string(), "\\u{2764}\\u{a}\\u{21}");
    /// ```
    #[must_use = "this returns the escaped string as an iterator, \
                  without modifying the original"]
    #[stable(feature = "str_escape", since = "1.34.0")]
    pub fn escape_unicode(&self) -> EscapeUnicode<'_> {
        EscapeUnicode { inner: self.chars().flat_map(CharEscapeUnicode) }
    }

    /// Returns the range that a substring points to.
    ///
    /// Returns `None` if `substr` does not point within `self`.
    ///
    /// Unlike [`str::find`], **this does not search through the string**.
    /// Instead, it uses pointer arithmetic to find where in the string
    /// `substr` is derived from.
    ///
    /// This is useful for extending [`str::split`] and similar methods.
    ///
    /// Note that this method may return false positives (typically either
    /// `Some(0..0)` or `Some(self.len()..self.len())`) if `substr` is a
    /// zero-length `str` that points at the beginning or end of another,
    /// independent, `str`.
    ///
    /// # Examples
    /// ```
    /// #![feature(substr_range)]
    ///
    /// let data = "a, b, b, a";
    /// let mut iter = data.split(", ").map(|s| data.substr_range(s).unwrap());
    ///
    /// assert_eq!(iter.next(), Some(0..1));
    /// assert_eq!(iter.next(), Some(3..4));
    /// assert_eq!(iter.next(), Some(6..7));
    /// assert_eq!(iter.next(), Some(9..10));
    /// ```
    #[must_use]
    #[unstable(feature = "substr_range", issue = "126769")]
    pub fn substr_range(&self, substr: &str) -> Option<Range<usize>> {
        self.as_bytes().subslice_range(substr.as_bytes())
    }

    /// Returns the same string as a string slice `&str`.
    ///
    /// This method is redundant when used directly on `&str`, but
    /// it helps dereferencing other string-like types to string slices,
    /// for example references to `Box<str>` or `Arc<str>`.
    #[inline]
    #[unstable(feature = "str_as_str", issue = "130366")]
    pub fn as_str(&self) -> &str {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<[u8]> for str {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for &str {
    /// Creates an empty str
    #[inline]
    fn default() -> Self {
        ""
    }
}

#[stable(feature = "default_mut_str", since = "1.28.0")]
impl Default for &mut str {
    /// Creates an empty mutable str
    #[inline]
    fn default() -> Self {
        // SAFETY: The empty string is valid UTF-8.
        unsafe { from_utf8_unchecked_mut(&mut []) }
    }
}

impl_fn_for_zst! {
    /// A nameable, cloneable fn type
    #[derive(Clone)]
    struct LinesMap impl<'a> Fn = |line: &'a str| -> &'a str {
        let Some(line) = line.strip_suffix('\n') else { return line };
        let Some(line) = line.strip_suffix('\r') else { return line };
        line
    };

    #[derive(Clone)]
    struct CharEscapeDebugContinue impl Fn = |c: char| -> char::EscapeDebug {
        c.escape_debug_ext(EscapeDebugExtArgs {
            escape_grapheme_extended: false,
            escape_single_quote: true,
            escape_double_quote: true
        })
    };

    #[derive(Clone)]
    struct CharEscapeUnicode impl Fn = |c: char| -> char::EscapeUnicode {
        c.escape_unicode()
    };
    #[derive(Clone)]
    struct CharEscapeDefault impl Fn = |c: char| -> char::EscapeDefault {
        c.escape_default()
    };

    #[derive(Clone)]
    struct IsWhitespace impl Fn = |c: char| -> bool {
        c.is_whitespace()
    };

    #[derive(Clone)]
    struct IsAsciiWhitespace impl Fn = |byte: &u8| -> bool {
        byte.is_ascii_whitespace()
    };

    #[derive(Clone)]
    struct IsNotEmpty impl<'a, 'b> Fn = |s: &'a &'b str| -> bool {
        !s.is_empty()
    };

    #[derive(Clone)]
    struct BytesIsNotEmpty impl<'a, 'b> Fn = |s: &'a &'b [u8]| -> bool {
        !s.is_empty()
    };

    #[derive(Clone)]
    struct UnsafeBytesToStr impl<'a> Fn = |bytes: &'a [u8]| -> &'a str {
        // SAFETY: not safe
        unsafe { from_utf8_unchecked(bytes) }
    };
}

// This is required to make `impl From<&str> for Box<dyn Error>` and `impl<E> From<E> for Box<dyn Error>` not overlap.
#[stable(feature = "error_in_core_neg_impl", since = "1.65.0")]
impl !crate::error::Error for &str {}
