//! A UTF-8–encoded, growable string, with allocator support.
//!
//! This module only exists due to type parameter defaults not affecting inference,
//! otherwise, [`alloc::string::String`][crate::string::String] would be the canonical location of `String`,
//! instead of a type alias.

#![unstable(feature = "string_allocator_api", issue = "32838")]

use core::cmp::Ordering;
use core::error::Error;
use core::fmt;
use core::hash;
#[cfg(not(no_global_oom_handling))]
use core::iter::from_fn;
use core::iter::FusedIterator;
#[cfg(not(no_global_oom_handling))]
use core::ops::Bound::{Excluded, Included, Unbounded};
use core::ops::{self, Range, RangeBounds};
#[cfg(not(no_global_oom_handling))]
use core::ops::{Add, AddAssign};
use core::ptr;
use core::slice;
use core::str::pattern::Pattern;

use crate::alloc::{Allocator, Global};
#[cfg(not(no_global_oom_handling))]
use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
use crate::collections::TryReserveError;
use crate::str::{self, from_utf8_unchecked_mut, Chars, Utf8Error};
#[cfg(not(no_global_oom_handling))]
use crate::str::{from_boxed_utf8_unchecked, FromStr};
use crate::string::ToString;
use crate::vec::Vec;

/// A UTF-8–encoded, growable string, with allocator support.
///
/// The documentation for this type is located at [`alloc::string::String`].
///
/// This is due to type parameter defaults not affecting inference.
/// Otherwise, [`alloc::string::String`] would be the canonical location of `String`,
/// instead of a type alias.
#[cfg_attr(not(test), lang = "String")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct String<#[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global> {
    vec: Vec<u8, A>,
}

/// A possible error value when converting a `String` from a UTF-8 byte vector.
///
/// This type is the error type for the [`from_utf8_in`] method on [`String`]. It
/// is designed in such a way to carefully avoid reallocations: the
/// [`into_bytes`] method will give back the byte vector that was used in the
/// conversion attempt.
///
/// [`from_utf8`]: String::from_utf8
/// [`into_bytes`]: FromUtf8Error::into_bytes
///
/// The [`Utf8Error`] type provided by [`std::str`] represents an error that may
/// occur when converting a slice of [`u8`]s to a [`&str`]. In this sense, it's
/// an analogue to `FromUtf8Error`, and you can get one from a `FromUtf8Error`
/// through the [`utf8_error`] method.
///
/// [`Utf8Error`]: str::Utf8Error "std::str::Utf8Error"
/// [`std::str`]: core::str "std::str"
/// [`&str`]: prim@str "&str"
/// [`utf8_error`]: FromUtf8Error::utf8_error
///
/// # Examples
///
/// ```
/// // some invalid bytes, in a vector
/// let bytes = vec![0, 159];
///
/// let value = String::from_utf8(bytes);
///
/// assert!(value.is_err());
/// assert_eq!(vec![0, 159], value.unwrap_err().into_bytes());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(no_global_oom_handling), derive(Clone))]
pub struct FromUtf8Error<
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    bytes: Vec<u8, A>,
    error: Utf8Error,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Eq for FromUtf8Error<A> {} // FIXME(zachs18): Structural(Partial)Eq?

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> PartialEq for FromUtf8Error<A> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.error == other.error
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> fmt::Debug for FromUtf8Error<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FromUtf8Error")
            .field("bytes", &self.bytes)
            .field("error", &self.error)
            .finish()
    }
}

/// A possible error value when converting a `String` from a UTF-16 byte slice.
///
/// This type is the error type for the [`from_utf16`] method on [`String`].
///
/// [`from_utf16`]: String::from_utf16
///
/// # Examples
///
/// ```
/// // 𝄞mu<invalid>ic
/// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075,
///           0xD800, 0x0069, 0x0063];
///
/// assert!(String::from_utf16(v).is_err());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct FromUtf16Error(());

impl String {
    /// Creates a new empty `String`.
    ///
    /// Given that the `String` is empty, this will not allocate any initial
    /// buffer. While that means that this initial operation is very
    /// inexpensive, it may cause excessive allocation later when you add
    /// data. If you have an idea of how much data the `String` will hold,
    /// consider the [`with_capacity`] method to prevent excessive
    /// re-allocation.
    ///
    /// [`with_capacity`]: String::with_capacity
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::new();
    /// ```
    #[inline]
    #[rustc_const_stable(feature = "const_string_new", since = "1.39.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "string_new")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub const fn new() -> String {
        String { vec: Vec::new() }
    }

    /// Creates a new empty `String` with at least the specified capacity.
    ///
    /// `String`s have an internal buffer to hold their data. The capacity is
    /// the length of that buffer, and can be queried with the [`capacity`]
    /// method. This method creates an empty `String`, but one with an initial
    /// buffer that can hold at least `capacity` bytes. This is useful when you
    /// may be appending a bunch of data to the `String`, reducing the number of
    /// reallocations it needs to do.
    ///
    /// [`capacity`]: String::capacity
    ///
    /// If the given capacity is `0`, no allocation will occur, and this method
    /// is identical to the [`new`] method.
    ///
    /// [`new`]: String::new
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::with_capacity(10);
    ///
    /// // The String contains no chars, even though it has capacity for more
    /// assert_eq!(s.len(), 0);
    ///
    /// // These are all done without reallocating...
    /// let cap = s.capacity();
    /// for _ in 0..10 {
    ///     s.push('a');
    /// }
    ///
    /// assert_eq!(s.capacity(), cap);
    ///
    /// // ...but this may make the string reallocate
    /// s.push('a');
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> String {
        String { vec: Vec::with_capacity(capacity) }
    }

    /// Creates a new empty `String` with at least the specified capacity.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the capacity exceeds `isize::MAX` bytes,
    /// or if the memory allocator reports failure.
    ///
    #[inline]
    #[unstable(feature = "try_with_capacity", issue = "91913")]
    pub fn try_with_capacity(capacity: usize) -> Result<String, TryReserveError> {
        Ok(String { vec: Vec::try_with_capacity(capacity)? })
    }

    // HACK(japaric): with cfg(test) the inherent `[T]::to_vec` method, which is
    // required for this method definition, is not available. Since we don't
    // require this method for testing purposes, I'll just stub it
    // NB see the slice::hack module in slice.rs for more information
    #[inline]
    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn from_str(_: &str) -> String {
        panic!("not available with cfg(test)");
    }
}

impl<A: Allocator> String<A> {
    /// Creates a new empty `String<A>` with the provided allocator.
    ///
    /// Given that the `String` is empty, this will not allocate any initial
    /// buffer. While that means that this initial operation is very
    /// inexpensive, it may cause excessive allocation later when you add
    /// data. If you have an idea of how much data the `String` will hold,
    /// consider the [`with_capacity_in`] method to prevent excessive
    /// re-allocation.
    ///
    /// [`with_capacity_in`]: String::with_capacity_in
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(string_allocator_api)]
    /// #![feature(allocator_api)]
    ///
    /// use std::string::string::String;
    /// use std::alloc::System;
    ///
    /// let s = String::new_in(System);
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[must_use]
    pub const fn new_in(alloc: A) -> String<A> {
        String { vec: Vec::new_in(alloc) }
    }

    /// Creates a new empty `String` with at least the specified capacity with the provided allocator.
    ///
    /// `String`s have an internal buffer to hold their data. The capacity is
    /// the length of that buffer, and can be queried with the [`capacity`]
    /// method. This method creates an empty `String`, but one with an initial
    /// buffer that can hold at least `capacity` bytes. This is useful when you
    /// may be appending a bunch of data to the `String`, reducing the number of
    /// reallocations it needs to do.
    ///
    /// [`capacity`]: String::capacity
    ///
    /// If the given capacity is `0`, no allocation will occur, and this method
    /// is identical to the [`new_in`] method.
    ///
    /// [`new_in`]: String::new_in
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(string_allocator_api)]
    /// #![feature(allocator_api)]
    ///
    /// use std::string::string::String;
    /// use std::alloc::System;
    ///
    /// let mut s = String::with_capacity_in(10, System);
    ///
    /// // The String contains no chars, even though it has capacity for more
    /// assert_eq!(s.len(), 0);
    ///
    /// // These are all done without reallocating...
    /// let cap = s.capacity();
    /// for _ in 0..10 {
    ///     s.push('a');
    /// }
    ///
    /// assert_eq!(s.capacity(), cap);
    ///
    /// // ...but this may make the string reallocate
    /// s.push('a');
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[must_use]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> String<A> {
        String { vec: Vec::with_capacity_in(capacity, alloc) }
    }

    /// Creates a new empty `String<A>` with at least the specified capacity with the provided allocator.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the capacity exceeds `isize::MAX` bytes,
    /// or if the memory allocator reports failure.
    ///
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "try_with_capacity", issue = "91913")]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<String<A>, TryReserveError> {
        Ok(String { vec: Vec::try_with_capacity_in(capacity, alloc)? })
    }

    /// Converts a vector of bytes to a `String`.
    ///
    /// A string ([`String`]) is made of bytes ([`u8`]), and a vector of bytes
    /// ([`Vec<u8>`]) is made of bytes, so this function converts between the
    /// two. Not all byte slices are valid `String`s, however: `String`
    /// requires that it is valid UTF-8. `from_utf8()` checks to ensure that
    /// the bytes are valid UTF-8, and then does the conversion.
    ///
    /// If you are sure that the byte slice is valid UTF-8, and you don't want
    /// to incur the overhead of the validity check, there is an unsafe version
    /// of this function, [`from_utf8_unchecked`], which has the same behavior
    /// but skips the check.
    ///
    /// This method will take care to not copy the vector, for efficiency's
    /// sake.
    ///
    /// If you need a [`&str`] instead of a `String`, consider
    /// [`str::from_utf8`].
    ///
    /// The inverse of this method is [`into_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the slice is not UTF-8 with a description as to why the
    /// provided bytes are not UTF-8. The vector you moved in is also included.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// // We know these bytes are valid, so we'll use `unwrap()`.
    /// let sparkle_heart = String::from_utf8(sparkle_heart).unwrap();
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    ///
    /// Incorrect bytes:
    ///
    /// ```
    /// // some invalid bytes, in a vector
    /// let sparkle_heart = vec![0, 159, 146, 150];
    ///
    /// assert!(String::from_utf8(sparkle_heart).is_err());
    /// ```
    ///
    /// See the docs for [`FromUtf8Error`] for more details on what you can do
    /// with this error.
    ///
    /// [`from_utf8_unchecked`]: String::from_utf8_unchecked
    /// [`Vec<u8>`]: crate::vec::Vec "Vec"
    /// [`&str`]: prim@str "&str"
    /// [`into_bytes`]: String::into_bytes
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "string_from_utf8")]
    pub fn from_utf8(vec: Vec<u8, A>) -> Result<String<A>, FromUtf8Error<A>> {
        match str::from_utf8(&vec) {
            Ok(..) => Ok(String { vec }),
            Err(e) => Err(FromUtf8Error { bytes: vec, error: e }),
        }
    }

    /// Decomposes a `String` into its raw components.
    ///
    /// Returns the raw pointer to the underlying data, the length of
    /// the string (in bytes), the allocated capacity of the data
    /// (in bytes), and the allocator. These are the same arguments in the same order as
    /// the arguments to [`from_raw_parts_in`].
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `String`. The only way to do
    /// this is to convert the raw pointer, length, and capacity back
    /// into a `String` with the [`from_raw_parts_in`] function, allowing
    /// the destructor to perform the cleanup.
    ///
    /// [`from_raw_parts_in`]: String::from_raw_parts_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(vec_into_raw_parts)]
    /// #![feature(string_allocator_api)]
    /// #![feature(allocator_api)]
    /// use std::string::string::String;
    /// use std::alloc::System;
    /// let mut s = String::new_in(System);
    /// s.push_str("hello");
    ///
    /// let (ptr, len, cap, alloc) = s.into_raw_parts_with_alloc();
    ///
    /// let rebuilt = unsafe { String::from_raw_parts_in(ptr, len, cap, alloc) };
    /// assert_eq!(rebuilt, "hello");
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "vec_into_raw_parts", reason = "new API", issue = "65816")]
    pub fn into_raw_parts_with_alloc(self) -> (*mut u8, usize, usize, A) {
        self.vec.into_raw_parts_with_alloc()
    }

    /// Creates a new `String` from a length, capacity, pointer, and allocator.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * The memory at `buf` needs to have been previously allocated by the
    ///   allocator `alloc`, with a required alignment of exactly 1.
    /// * `length` needs to be less than or equal to `capacity`.
    /// * `capacity` needs to be the correct value.
    /// * The first `length` bytes at `buf` need to be valid UTF-8.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures. For example, it is normally **not** safe to
    /// build a `String` from a pointer to a C `char` array containing UTF-8
    /// _unless_ you are certain that array was originally allocated by `alloc`.
    ///
    /// The ownership of `buf` is effectively transferred to the
    /// `String` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(string_allocator_api)]
    /// #![feature(allocator_api)]
    ///
    /// use std::string::string::String;
    /// use std::mem;
    /// use std::alloc::System;
    ///
    /// unsafe {
    ///     let mut s = String::new_in(System);
    ///     s.push_str("hello");
    ///
    // FIXME Update this when vec_into_raw_parts is stabilized
    ///     // Prevent automatically dropping the String's data
    ///     let mut s = mem::ManuallyDrop::new(s);
    ///
    ///     let ptr = s.as_mut_ptr();
    ///     let len = s.len();
    ///     let capacity = s.capacity();
    ///     let alloc = s.allocator().clone();
    ///
    ///     let s = String::from_raw_parts_in(ptr, len, capacity, alloc);
    ///
    ///     assert_eq!("hello", s);
    /// }
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub unsafe fn from_raw_parts_in(
        buf: *mut u8,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> String<A> {
        unsafe { String { vec: Vec::from_raw_parts_in(buf, length, capacity, alloc) } }
    }

    /// Returns a reference to the underlying allocator.
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn allocator(&self) -> &A {
        self.vec.allocator()
    }
}

impl String {
    /// Converts a slice of bytes to a string, including invalid characters.
    ///
    /// Strings are made of bytes ([`u8`]), and a slice of bytes
    /// ([`&[u8]`][byteslice]) is made of bytes, so this function converts
    /// between the two. Not all byte slices are valid strings, however: strings
    /// are required to be valid UTF-8. During this conversion,
    /// `from_utf8_lossy()` will replace any invalid UTF-8 sequences with
    /// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD], which looks like this: �
    ///
    /// [byteslice]: prim@slice
    /// [U+FFFD]: core::char::REPLACEMENT_CHARACTER
    ///
    /// If you are sure that the byte slice is valid UTF-8, and you don't want
    /// to incur the overhead of the conversion, there is an unsafe version
    /// of this function, [`from_utf8_unchecked`], which has the same behavior
    /// but skips the checks.
    ///
    /// [`from_utf8_unchecked`]: String::from_utf8_unchecked
    ///
    /// This function returns a [`Cow<'a, str>`]. If our byte slice is invalid
    /// UTF-8, then we need to insert the replacement characters, which will
    /// change the size of the string, and hence, require a `String`. But if
    /// it's already valid UTF-8, we don't need a new allocation. This return
    /// type allows us to handle both cases.
    ///
    /// [`Cow<'a, str>`]: crate::borrow::Cow "borrow::Cow"
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// let sparkle_heart = String::from_utf8_lossy(&sparkle_heart);
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    ///
    /// Incorrect bytes:
    ///
    /// ```
    /// // some invalid bytes
    /// let input = b"Hello \xF0\x90\x80World";
    /// let output = String::from_utf8_lossy(input);
    ///
    /// assert_eq!("Hello �World", output);
    /// ```
    #[must_use]
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf8_lossy(v: &[u8]) -> Cow<'_, str> {
        match String::from_utf8_lossy_in(Global) {
            Ok(s) => Cow::Borrowed(s),
            Err(s) => Cow::Owned(s),
        }
    }

    fn from_utf8_lossy_in(v: &[u8], alloc: A) -> Result<&str, String<A>> {
        let mut iter = v.utf8_chunks();

        let first_valid = if let Some(chunk) = iter.next() {
            let valid = chunk.valid();
            if chunk.invalid().is_empty() {
                debug_assert_eq!(valid.len(), v.len());
                return Ok(valid);
            }
            valid
        } else {
            return Ok("");
        };

        const REPLACEMENT: &str = "\u{FFFD}";

        let mut res = String::with_capacity_in(v.len(), alloc);
        res.push_str(first_valid);
        res.push_str(REPLACEMENT);

        for chunk in iter {
            res.push_str(chunk.valid());
            if !chunk.invalid().is_empty() {
                res.push_str(REPLACEMENT);
            }
        }

        Err(res)
    }

    /// Converts a [`Vec<u8>`] to a `String`, substituting invalid UTF-8
    /// sequences with replacement characters.
    ///
    /// See [`from_utf8_lossy`] for more details.
    ///
    /// [`from_utf8_lossy`]: String::from_utf8_lossy
    ///
    /// Note that this function does not guarantee reuse of the original `Vec`
    /// allocation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(string_from_utf8_lossy_owned)]
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// let sparkle_heart = String::from_utf8_lossy_owned(sparkle_heart);
    ///
    /// assert_eq!(String::from("💖"), sparkle_heart);
    /// ```
    ///
    /// Incorrect bytes:
    ///
    /// ```
    /// #![feature(string_from_utf8_lossy_owned)]
    /// // some invalid bytes
    /// let input: Vec<u8> = b"Hello \xF0\x90\x80World".into();
    /// let output = String::from_utf8_lossy_owned(input);
    ///
    /// assert_eq!(String::from("Hello �World"), output);
    /// ```
    #[must_use]
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "string_from_utf8_lossy_owned", issue = "129436")]
    pub fn from_utf8_lossy_owned(v: Vec<u8>) -> String {
        if let Cow::Owned(string) = String::from_utf8_lossy(&v) {
            string
        } else {
            // SAFETY: `String::from_utf8_lossy`'s contract ensures that if
            // it returns a `Cow::Borrowed`, it is a valid UTF-8 string.
            // Otherwise, it returns a new allocation of an owned `String`, with
            // replacement characters for invalid sequences, which is returned
            // above.
            unsafe { String::from_utf8_unchecked(v) }
        }
    }


    /// Decode a native endian UTF-16–encoded vector `v` into a `String`,
    /// returning [`Err`] if `v` contains any invalid data.
    ///
    /// # Examples
    ///
    /// ```
    /// // 𝄞music
    /// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075,
    ///           0x0073, 0x0069, 0x0063];
    /// assert_eq!(String::from("𝄞music"),
    ///            String::from_utf16(v).unwrap());
    ///
    /// // 𝄞mu<invalid>ic
    /// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075,
    ///           0xD800, 0x0069, 0x0063];
    /// assert!(String::from_utf16(v).is_err());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf16(v: &[u16]) -> Result<String, FromUtf16Error> {
        // This isn't done via collect::<Result<_, _>>() for performance reasons.
        // FIXME: the function can be simplified again when #48994 is closed.
        let mut ret = String::with_capacity(v.len());
        for c in char::decode_utf16(v.iter().cloned()) {
            if let Ok(c) = c {
                ret.push(c);
            } else {
                return Err(FromUtf16Error(()));
            }
        }
        Ok(ret)
    }

    /// Decode a native endian UTF-16–encoded slice `v` into a `String`,
    /// replacing invalid data with [the replacement character (`U+FFFD`)][U+FFFD].
    ///
    /// Unlike [`from_utf8_lossy`] which returns a [`Cow<'a, str>`],
    /// `from_utf16_lossy` returns a `String` since the UTF-16 to UTF-8
    /// conversion requires a memory allocation.
    ///
    /// [`from_utf8_lossy`]: String::from_utf8_lossy
    /// [`Cow<'a, str>`]: crate::borrow::Cow "borrow::Cow"
    /// [U+FFFD]: core::char::REPLACEMENT_CHARACTER
    ///
    /// # Examples
    ///
    /// ```
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075,
    ///           0x0073, 0xDD1E, 0x0069, 0x0063,
    ///           0xD834];
    ///
    /// assert_eq!(String::from("𝄞mus\u{FFFD}ic\u{FFFD}"),
    ///            String::from_utf16_lossy(v));
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[must_use]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf16_lossy(v: &[u16]) -> String {
        char::decode_utf16(v.iter().cloned())
            .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect()
    }

    /// Decode a UTF-16LE–encoded vector `v` into a `String`,
    /// returning [`Err`] if `v` contains any invalid data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(str_from_utf16_endian)]
    /// // 𝄞music
    /// let v = &[0x34, 0xD8, 0x1E, 0xDD, 0x6d, 0x00, 0x75, 0x00,
    ///           0x73, 0x00, 0x69, 0x00, 0x63, 0x00];
    /// assert_eq!(String::from("𝄞music"),
    ///            String::from_utf16le(v).unwrap());
    ///
    /// // 𝄞mu<invalid>ic
    /// let v = &[0x34, 0xD8, 0x1E, 0xDD, 0x6d, 0x00, 0x75, 0x00,
    ///           0x00, 0xD8, 0x69, 0x00, 0x63, 0x00];
    /// assert!(String::from_utf16le(v).is_err());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "str_from_utf16_endian", issue = "116258")]
    pub fn from_utf16le(v: &[u8]) -> Result<String, FromUtf16Error> {
        if v.len() % 2 != 0 {
            return Err(FromUtf16Error(()));
        }
        match (cfg!(target_endian = "little"), unsafe { v.align_to::<u16>() }) {
            (true, ([], v, [])) => Self::from_utf16(v),
            _ => char::decode_utf16(v.array_chunks::<2>().copied().map(u16::from_le_bytes))
                .collect::<Result<_, _>>()
                .map_err(|_| FromUtf16Error(())),
        }
    }

    /// Decode a UTF-16LE–encoded slice `v` into a `String`, replacing
    /// invalid data with [the replacement character (`U+FFFD`)][U+FFFD].
    ///
    /// Unlike [`from_utf8_lossy`] which returns a [`Cow<'a, str>`],
    /// `from_utf16le_lossy` returns a `String` since the UTF-16 to UTF-8
    /// conversion requires a memory allocation.
    ///
    /// [`from_utf8_lossy`]: String::from_utf8_lossy
    /// [`Cow<'a, str>`]: crate::borrow::Cow "borrow::Cow"
    /// [U+FFFD]: core::char::REPLACEMENT_CHARACTER
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(str_from_utf16_endian)]
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = &[0x34, 0xD8, 0x1E, 0xDD, 0x6d, 0x00, 0x75, 0x00,
    ///           0x73, 0x00, 0x1E, 0xDD, 0x69, 0x00, 0x63, 0x00,
    ///           0x34, 0xD8];
    ///
    /// assert_eq!(String::from("𝄞mus\u{FFFD}ic\u{FFFD}"),
    ///            String::from_utf16le_lossy(v));
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "str_from_utf16_endian", issue = "116258")]
    pub fn from_utf16le_lossy(v: &[u8]) -> String {
        match (cfg!(target_endian = "little"), unsafe { v.align_to::<u16>() }) {
            (true, ([], v, [])) => Self::from_utf16_lossy(v),
            (true, ([], v, [_remainder])) => Self::from_utf16_lossy(v) + "\u{FFFD}",
            _ => {
                let mut iter = v.array_chunks::<2>();
                let string = char::decode_utf16(iter.by_ref().copied().map(u16::from_le_bytes))
                    .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
                    .collect();
                if iter.remainder().is_empty() { string } else { string + "\u{FFFD}" }
            }
        }
    }

    /// Decode a UTF-16BE–encoded vector `v` into a `String`,
    /// returning [`Err`] if `v` contains any invalid data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(str_from_utf16_endian)]
    /// // 𝄞music
    /// let v = &[0xD8, 0x34, 0xDD, 0x1E, 0x00, 0x6d, 0x00, 0x75,
    ///           0x00, 0x73, 0x00, 0x69, 0x00, 0x63];
    /// assert_eq!(String::from("𝄞music"),
    ///            String::from_utf16be(v).unwrap());
    ///
    /// // 𝄞mu<invalid>ic
    /// let v = &[0xD8, 0x34, 0xDD, 0x1E, 0x00, 0x6d, 0x00, 0x75,
    ///           0xD8, 0x00, 0x00, 0x69, 0x00, 0x63];
    /// assert!(String::from_utf16be(v).is_err());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "str_from_utf16_endian", issue = "116258")]
    pub fn from_utf16be(v: &[u8]) -> Result<String, FromUtf16Error> {
        if v.len() % 2 != 0 {
            return Err(FromUtf16Error(()));
        }
        match (cfg!(target_endian = "big"), unsafe { v.align_to::<u16>() }) {
            (true, ([], v, [])) => Self::from_utf16(v),
            _ => char::decode_utf16(v.array_chunks::<2>().copied().map(u16::from_be_bytes))
                .collect::<Result<_, _>>()
                .map_err(|_| FromUtf16Error(())),
        }
    }

    /// Decode a UTF-16BE–encoded slice `v` into a `String`, replacing
    /// invalid data with [the replacement character (`U+FFFD`)][U+FFFD].
    ///
    /// Unlike [`from_utf8_lossy`] which returns a [`Cow<'a, str>`],
    /// `from_utf16le_lossy` returns a `String` since the UTF-16 to UTF-8
    /// conversion requires a memory allocation.
    ///
    /// [`from_utf8_lossy`]: String::from_utf8_lossy
    /// [`Cow<'a, str>`]: crate::borrow::Cow "borrow::Cow"
    /// [U+FFFD]: core::char::REPLACEMENT_CHARACTER
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(str_from_utf16_endian)]
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = &[0xD8, 0x34, 0xDD, 0x1E, 0x00, 0x6d, 0x00, 0x75,
    ///           0x00, 0x73, 0xDD, 0x1E, 0x00, 0x69, 0x00, 0x63,
    ///           0xD8, 0x34];
    ///
    /// assert_eq!(String::from("𝄞mus\u{FFFD}ic\u{FFFD}"),
    ///            String::from_utf16be_lossy(v));
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "str_from_utf16_endian", issue = "116258")]
    pub fn from_utf16be_lossy(v: &[u8]) -> String {
        match (cfg!(target_endian = "big"), unsafe { v.align_to::<u16>() }) {
            (true, ([], v, [])) => Self::from_utf16_lossy(v),
            (true, ([], v, [_remainder])) => Self::from_utf16_lossy(v) + "\u{FFFD}",
            _ => {
                let mut iter = v.array_chunks::<2>();
                let string = char::decode_utf16(iter.by_ref().copied().map(u16::from_be_bytes))
                    .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
                    .collect();
                if iter.remainder().is_empty() { string } else { string + "\u{FFFD}" }
            }
        }
    }

    /// Decomposes a `String` into its raw components: `(pointer, length, capacity)`.
    ///
    /// Returns the raw pointer to the underlying data, the length of
    /// the string (in bytes), and the allocated capacity of the data
    /// (in bytes). These are the same arguments in the same order as
    /// the arguments to [`from_raw_parts`].
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `String`. The only way to do
    /// this is to convert the raw pointer, length, and capacity back
    /// into a `String` with the [`from_raw_parts`] function, allowing
    /// the destructor to perform the cleanup.
    ///
    /// [`from_raw_parts`]: String::from_raw_parts
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(vec_into_raw_parts)]
    /// let s = String::from("hello");
    ///
    /// let (ptr, len, cap) = s.into_raw_parts();
    ///
    /// let rebuilt = unsafe { String::from_raw_parts(ptr, len, cap) };
    /// assert_eq!(rebuilt, "hello");
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[unstable(feature = "vec_into_raw_parts", reason = "new API", issue = "65816")]
    pub fn into_raw_parts(self) -> (*mut u8, usize, usize) {
        self.vec.into_raw_parts()
    }

    /// Creates a new `String` from a pointer, a length and a capacity.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * The memory at `buf` needs to have been previously allocated by the
    ///   same allocator the standard library uses, with a required alignment of exactly 1.
    /// * `length` needs to be less than or equal to `capacity`.
    /// * `capacity` needs to be the correct value.
    /// * The first `length` bytes at `buf` need to be valid UTF-8.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures. For example, it is normally **not** safe to
    /// build a `String` from a pointer to a C `char` array containing UTF-8
    /// _unless_ you are certain that array was originally allocated by the
    /// Rust standard library's allocator.
    ///
    /// The ownership of `buf` is effectively transferred to the
    /// `String` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem;
    ///
    /// unsafe {
    ///     let s = String::from("hello");
    ///
    // FIXME Update this when vec_into_raw_parts is stabilized
    ///     // Prevent automatically dropping the String's data
    ///     let mut s = mem::ManuallyDrop::new(s);
    ///
    ///     let ptr = s.as_mut_ptr();
    ///     let len = s.len();
    ///     let capacity = s.capacity();
    ///
    ///     let s = String::from_raw_parts(ptr, len, capacity);
    ///
    ///     assert_eq!(String::from("hello"), s);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_raw_parts(buf: *mut u8, length: usize, capacity: usize) -> String {
        unsafe { String { vec: Vec::from_raw_parts(buf, length, capacity) } }
    }
}

impl<A: Allocator> String<A> {
    /// Converts a vector of bytes to a `String` without checking that the
    /// string contains valid UTF-8.
    ///
    /// See the safe version, [`from_utf8`], for more details.
    ///
    /// [`from_utf8`]: String::from_utf8
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check that the bytes passed
    /// to it are valid UTF-8. If this constraint is violated, it may cause
    /// memory unsafety issues with future users of the `String`, as the rest of
    /// the standard library assumes that `String`s are valid UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// let sparkle_heart = unsafe {
    ///     String::from_utf8_unchecked(sparkle_heart)
    /// };
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_utf8_unchecked(bytes: Vec<u8, A>) -> String<A> {
        String { vec: bytes }
    }

    /// Converts a `String` into a byte vector.
    ///
    /// This consumes the `String`, so we do not need to copy its contents.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from("hello");
    /// let bytes = s.into_bytes();
    ///
    /// assert_eq!(&[104, 101, 108, 108, 111][..], &bytes[..]);
    /// ```
    #[inline]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    pub const fn into_bytes(self) -> Vec<u8, A> {
        self.vec
    }

    /// Extracts a string slice containing the entire `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from("foo");
    ///
    /// assert_eq!("foo", s.as_str());
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "string_as_str", since = "1.7.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "string_as_str")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    pub const fn as_str(&self) -> &str {
        // SAFETY: String contents are stipulated to be valid UTF-8, invalid contents are an error
        // at construction.
        unsafe { str::from_utf8_unchecked(self.vec.as_slice()) }
    }

    /// Converts a `String` into a mutable string slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foobar");
    /// let s_mut_str = s.as_mut_str();
    ///
    /// s_mut_str.make_ascii_uppercase();
    ///
    /// assert_eq!("FOOBAR", s_mut_str);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "string_as_str", since = "1.7.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "string_as_mut_str")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    pub const fn as_mut_str(&mut self) -> &mut str {
        // SAFETY: String contents are stipulated to be valid UTF-8, invalid contents are an error
        // at construction.
        unsafe { str::from_utf8_unchecked_mut(self.vec.as_mut_slice()) }
    }

    /// Appends a given string slice onto the end of this `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    ///
    /// s.push_str("bar");
    ///
    /// assert_eq!("foobar", s);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_confusables("append", "push")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "string_push_str")]
    pub fn push_str(&mut self, string: &str) {
        self.vec.extend_from_slice(string.as_bytes())
    }

    /// Copies elements from `src` range to the end of the string.
    ///
    /// # Panics
    ///
    /// Panics if the starting point or end point do not lie on a [`char`]
    /// boundary, or if they're out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(string_extend_from_within)]
    /// let mut string = String::from("abcde");
    ///
    /// string.extend_from_within(2..);
    /// assert_eq!(string, "abcdecde");
    ///
    /// string.extend_from_within(..2);
    /// assert_eq!(string, "abcdecdeab");
    ///
    /// string.extend_from_within(4..8);
    /// assert_eq!(string, "abcdecdeabecde");
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "string_extend_from_within", issue = "103806")]
    pub fn extend_from_within<R>(&mut self, src: R)
    where
        R: RangeBounds<usize>,
    {
        let src @ Range { start, end } = slice::range(src, ..self.len());

        assert!(self.is_char_boundary(start));
        assert!(self.is_char_boundary(end));

        self.vec.extend_from_within(src);
    }

    /// Returns this `String`'s capacity, in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::with_capacity(10);
    ///
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    pub const fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    /// Reserves capacity for at least `additional` bytes more than the
    /// current length. The allocator may reserve more space to speculatively
    /// avoid frequent allocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows [`usize`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = String::new();
    ///
    /// s.reserve(10);
    ///
    /// assert!(s.capacity() >= 10);
    /// ```
    ///
    /// This might not actually increase the capacity:
    ///
    /// ```
    /// let mut s = String::with_capacity(10);
    /// s.push('a');
    /// s.push('b');
    ///
    /// // s now has a length of 2 and a capacity of at least 10
    /// let capacity = s.capacity();
    /// assert_eq!(2, s.len());
    /// assert!(capacity >= 10);
    ///
    /// // Since we already have at least an extra 8 capacity, calling this...
    /// s.reserve(8);
    ///
    /// // ... doesn't actually increase.
    /// assert_eq!(capacity, s.capacity());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        self.vec.reserve(additional)
    }

    /// Reserves the minimum capacity for at least `additional` bytes more than
    /// the current length. Unlike [`reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// [`reserve`]: String::reserve
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows [`usize`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = String::new();
    ///
    /// s.reserve_exact(10);
    ///
    /// assert!(s.capacity() >= 10);
    /// ```
    ///
    /// This might not actually increase the capacity:
    ///
    /// ```
    /// let mut s = String::with_capacity(10);
    /// s.push('a');
    /// s.push('b');
    ///
    /// // s now has a length of 2 and a capacity of at least 10
    /// let capacity = s.capacity();
    /// assert_eq!(2, s.len());
    /// assert!(capacity >= 10);
    ///
    /// // Since we already have at least an extra 8 capacity, calling this...
    /// s.reserve_exact(8);
    ///
    /// // ... doesn't actually increase.
    /// assert_eq!(capacity, s.capacity());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.vec.reserve_exact(additional)
    }

    /// Tries to reserve capacity for at least `additional` bytes more than the
    /// current length. The allocator may reserve more space to speculatively
    /// avoid frequent allocations. After calling `try_reserve`, capacity will be
    /// greater than or equal to `self.len() + additional` if it returns
    /// `Ok(())`. Does nothing if capacity is already sufficient. This method
    /// preserves the contents even if an error occurs.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TryReserveError;
    ///
    /// fn process_data(data: &str) -> Result<String, TryReserveError> {
    ///     let mut output = String::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     output.try_reserve(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     output.push_str(data);
    ///
    ///     Ok(output)
    /// }
    /// # process_data("rust").expect("why is the test harness OOMing on 4 bytes?");
    /// ```
    #[stable(feature = "try_reserve", since = "1.57.0")]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.vec.try_reserve(additional)
    }

    /// Tries to reserve the minimum capacity for at least `additional` bytes
    /// more than the current length. Unlike [`try_reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `try_reserve_exact`, capacity will be greater than or
    /// equal to `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`try_reserve`] if future insertions are expected.
    ///
    /// [`try_reserve`]: String::try_reserve
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TryReserveError;
    ///
    /// fn process_data(data: &str) -> Result<String, TryReserveError> {
    ///     let mut output = String::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     output.try_reserve_exact(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     output.push_str(data);
    ///
    ///     Ok(output)
    /// }
    /// # process_data("rust").expect("why is the test harness OOMing on 4 bytes?");
    /// ```
    #[stable(feature = "try_reserve", since = "1.57.0")]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.vec.try_reserve_exact(additional)
    }

    /// Shrinks the capacity of this `String` to match its length.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    ///
    /// s.reserve(100);
    /// assert!(s.capacity() >= 100);
    ///
    /// s.shrink_to_fit();
    /// assert_eq!(3, s.capacity());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit()
    }

    /// Shrinks the capacity of this `String` with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    ///
    /// s.reserve(100);
    /// assert!(s.capacity() >= 100);
    ///
    /// s.shrink_to(10);
    /// assert!(s.capacity() >= 10);
    /// s.shrink_to(0);
    /// assert!(s.capacity() >= 3);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "shrink_to", since = "1.56.0")]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.vec.shrink_to(min_capacity)
    }

    /// Appends the given [`char`] to the end of this `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("abc");
    ///
    /// s.push('1');
    /// s.push('2');
    /// s.push('3');
    ///
    /// assert_eq!("abc123", s);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, ch: char) {
        match ch.len_utf8() {
            1 => self.vec.push(ch as u8),
            _ => self.vec.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }

    /// Returns a byte slice of this `String`'s contents.
    ///
    /// The inverse of this method is [`from_utf8`].
    ///
    /// [`from_utf8`]: String::from_utf8
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from("hello");
    ///
    /// assert_eq!(&[104, 101, 108, 108, 111], s.as_bytes());
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    pub const fn as_bytes(&self) -> &[u8] {
        self.vec.as_slice()
    }

    /// Shortens this `String` to the specified length.
    ///
    /// If `new_len` is greater than or equal to the string's current length, this has no
    /// effect.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the string
    ///
    /// # Panics
    ///
    /// Panics if `new_len` does not lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("hello");
    ///
    /// s.truncate(2);
    ///
    /// assert_eq!("he", s);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len <= self.len() {
            assert!(self.is_char_boundary(new_len));
            self.vec.truncate(new_len)
        }
    }

    /// Removes the last character from the string buffer and returns it.
    ///
    /// Returns [`None`] if this `String` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("abč");
    ///
    /// assert_eq!(s.pop(), Some('č'));
    /// assert_eq!(s.pop(), Some('b'));
    /// assert_eq!(s.pop(), Some('a'));
    ///
    /// assert_eq!(s.pop(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.chars().rev().next()?;
        let newlen = self.len() - ch.len_utf8();
        unsafe {
            self.vec.set_len(newlen);
        }
        Some(ch)
    }

    /// Removes a [`char`] from this `String` at a byte position and returns it.
    ///
    /// This is an *O*(*n*) operation, as it requires copying every element in the
    /// buffer.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is larger than or equal to the `String`'s length,
    /// or if it does not lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("abç");
    ///
    /// assert_eq!(s.remove(0), 'a');
    /// assert_eq!(s.remove(1), 'ç');
    /// assert_eq!(s.remove(0), 'b');
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_confusables("delete", "take")]
    pub fn remove(&mut self, idx: usize) -> char {
        let ch = match self[idx..].chars().next() {
            Some(ch) => ch,
            None => panic!("cannot remove a char from the end of a string"),
        };

        let next = idx + ch.len_utf8();
        let len = self.len();
        unsafe {
            ptr::copy(self.vec.as_ptr().add(next), self.vec.as_mut_ptr().add(idx), len - next);
            self.vec.set_len(len - (next - idx));
        }
        ch
    }

    /// Remove all matches of pattern `pat` in the `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(string_remove_matches)]
    /// let mut s = String::from("Trees are not green, the sky is not blue.");
    /// s.remove_matches("not ");
    /// assert_eq!("Trees are green, the sky is blue.", s);
    /// ```
    ///
    /// Matches will be detected and removed iteratively, so in cases where
    /// patterns overlap, only the first pattern will be removed:
    ///
    /// ```
    /// #![feature(string_remove_matches)]
    /// let mut s = String::from("banana");
    /// s.remove_matches("ana");
    /// assert_eq!("bna", s);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "string_remove_matches", reason = "new API", issue = "72826")]
    pub fn remove_matches<P: Pattern>(&mut self, pat: P) {
        use core::str::pattern::Searcher;

        let rejections = {
            let mut searcher = pat.into_searcher(self);
            // Per Searcher::next:
            //
            // A Match result needs to contain the whole matched pattern,
            // however Reject results may be split up into arbitrary many
            // adjacent fragments. Both ranges may have zero length.
            //
            // In practice the implementation of Searcher::next_match tends to
            // be more efficient, so we use it here and do some work to invert
            // matches into rejections since that's what we want to copy below.
            let mut front = 0;
            let rejections: Vec<_> = from_fn(|| {
                let (start, end) = searcher.next_match()?;
                let prev_front = front;
                front = end;
                Some((prev_front, start))
            })
            .collect();
            rejections.into_iter().chain(core::iter::once((front, self.len())))
        };

        let mut len = 0;
        let ptr = self.vec.as_mut_ptr();

        for (start, end) in rejections {
            let count = end - start;
            if start != len {
                // SAFETY: per Searcher::next:
                //
                // The stream of Match and Reject values up to a Done will
                // contain index ranges that are adjacent, non-overlapping,
                // covering the whole haystack, and laying on utf8
                // boundaries.
                unsafe {
                    ptr::copy(ptr.add(start), ptr.add(len), count);
                }
            }
            len += count;
        }

        unsafe {
            self.vec.set_len(len);
        }
    }

    /// Retains only the characters specified by the predicate.
    ///
    /// In other words, remove all characters `c` such that `f(c)` returns `false`.
    /// This method operates in place, visiting each character exactly once in the
    /// original order, and preserves the order of the retained characters.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("f_o_ob_ar");
    ///
    /// s.retain(|c| c != '_');
    ///
    /// assert_eq!(s, "foobar");
    /// ```
    ///
    /// Because the elements are visited exactly once in the original order,
    /// external state may be used to decide which elements to keep.
    ///
    /// ```
    /// let mut s = String::from("abcde");
    /// let keep = [false, true, true, false, true];
    /// let mut iter = keep.iter();
    /// s.retain(|_| *iter.next().unwrap());
    /// assert_eq!(s, "bce");
    /// ```
    #[inline]
    #[stable(feature = "string_retain", since = "1.26.0")]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(char) -> bool,
    {
        struct SetLenOnDrop<'a, A: Allocator> {
            s: &'a mut String<A>,
            idx: usize,
            del_bytes: usize,
        }

        impl<'a, A: Allocator> Drop for SetLenOnDrop<'a, A> {
            fn drop(&mut self) {
                let new_len = self.idx - self.del_bytes;
                debug_assert!(new_len <= self.s.len());
                unsafe { self.s.vec.set_len(new_len) };
            }
        }

        let len = self.len();
        let mut guard = SetLenOnDrop { s: self, idx: 0, del_bytes: 0 };

        while guard.idx < len {
            let ch =
                // SAFETY: `guard.idx` is positive-or-zero and less that len so the `get_unchecked`
                // is in bound. `self` is valid UTF-8 like string and the returned slice starts at
                // a unicode code point so the `Chars` always return one character.
                unsafe { guard.s.get_unchecked(guard.idx..len).chars().next().unwrap_unchecked() };
            let ch_len = ch.len_utf8();

            if !f(ch) {
                guard.del_bytes += ch_len;
            } else if guard.del_bytes > 0 {
                // SAFETY: `guard.idx` is in bound and `guard.del_bytes` represent the number of
                // bytes that are erased from the string so the resulting `guard.idx -
                // guard.del_bytes` always represent a valid unicode code point.
                //
                // `guard.del_bytes` >= `ch.len_utf8()`, so taking a slice with `ch.len_utf8()` len
                // is safe.
                ch.encode_utf8(unsafe {
                    crate::slice::from_raw_parts_mut(
                        guard.s.as_mut_ptr().add(guard.idx - guard.del_bytes),
                        ch.len_utf8(),
                    )
                });
            }

            // Point idx to the next char
            guard.idx += ch_len;
        }

        drop(guard);
    }

    /// Inserts a character into this `String` at a byte position.
    ///
    /// This is an *O*(*n*) operation as it requires copying every element in the
    /// buffer.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is larger than the `String`'s length, or if it does not
    /// lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::with_capacity(3);
    ///
    /// s.insert(0, 'f');
    /// s.insert(1, 'o');
    /// s.insert(2, 'o');
    ///
    /// assert_eq!("foo", s);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_confusables("set")]
    pub fn insert(&mut self, idx: usize, ch: char) {
        assert!(self.is_char_boundary(idx));
        let mut bits = [0; 4];
        let bits = ch.encode_utf8(&mut bits).as_bytes();

        unsafe {
            self.insert_bytes(idx, bits);
        }
    }

    #[cfg(not(no_global_oom_handling))]
    unsafe fn insert_bytes(&mut self, idx: usize, bytes: &[u8]) {
        let len = self.len();
        let amt = bytes.len();
        self.vec.reserve(amt);

        unsafe {
            ptr::copy(self.vec.as_ptr().add(idx), self.vec.as_mut_ptr().add(idx + amt), len - idx);
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.vec.as_mut_ptr().add(idx), amt);
            self.vec.set_len(len + amt);
        }
    }

    /// Inserts a string slice into this `String` at a byte position.
    ///
    /// This is an *O*(*n*) operation as it requires copying every element in the
    /// buffer.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is larger than the `String`'s length, or if it does not
    /// lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("bar");
    ///
    /// s.insert_str(0, "foo");
    ///
    /// assert_eq!("foobar", s);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "insert_str", since = "1.16.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "string_insert_str")]
    pub fn insert_str(&mut self, idx: usize, string: &str) {
        assert!(self.is_char_boundary(idx));

        unsafe {
            self.insert_bytes(idx, string.as_bytes());
        }
    }

    /// Returns a mutable reference to the contents of this `String`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the returned `&mut Vec` allows writing
    /// bytes which are not valid UTF-8. If this constraint is violated, using
    /// the original `String` after dropping the `&mut Vec` may violate memory
    /// safety, as the rest of the standard library assumes that `String`s are
    /// valid UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("hello");
    ///
    /// unsafe {
    ///     let vec = s.as_mut_vec();
    ///     assert_eq!(&[104, 101, 108, 108, 111][..], &vec[..]);
    ///
    ///     vec.reverse();
    /// }
    /// assert_eq!(s, "olleh");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    pub const unsafe fn as_mut_vec(&mut self) -> &mut Vec<u8, A> {
        &mut self.vec
    }

    /// Returns the length of this `String`, in bytes, not [`char`]s or
    /// graphemes. In other words, it might not be what a human considers the
    /// length of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = String::from("foo");
    /// assert_eq!(a.len(), 3);
    ///
    /// let fancy_f = String::from("ƒoo");
    /// assert_eq!(fancy_f.len(), 4);
    /// assert_eq!(fancy_f.chars().count(), 3);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    #[rustc_confusables("length", "size")]
    pub const fn len(&self) -> usize {
        self.vec.len()
    }

    /// Returns `true` if this `String` has a length of zero, and `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::new();
    /// assert!(v.is_empty());
    ///
    /// v.push('a');
    /// assert!(!v.is_empty());
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_unstable(feature = "const_vec_string_slice", issue = "129041")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Splits the string into two at the given byte index.
    ///
    /// Returns a newly allocated `String`. `self` contains bytes `[0, at)`, and
    /// the returned `String` contains bytes `[at, len)`. `at` must be on the
    /// boundary of a UTF-8 code point.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// # Panics
    ///
    /// Panics if `at` is not on a `UTF-8` code point boundary, or if it is beyond the last
    /// code point of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// let mut hello = String::from("Hello, World!");
    /// let world = hello.split_off(7);
    /// assert_eq!(hello, "Hello, ");
    /// assert_eq!(world, "World!");
    /// # }
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "string_split_off", since = "1.16.0")]
    #[must_use = "use `.truncate()` if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> String<A>
    where
        A: Clone,
    {
        assert!(self.is_char_boundary(at));
        let other = self.vec.split_off(at);
        unsafe { String::from_utf8_unchecked(other) }
    }

    /// Truncates this `String`, removing all contents.
    ///
    /// While this means the `String` will have a length of zero, it does not
    /// touch its capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    ///
    /// s.clear();
    ///
    /// assert!(s.is_empty());
    /// assert_eq!(0, s.len());
    /// assert_eq!(3, s.capacity());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.vec.clear()
    }

    /// Removes the specified range from the string in bulk, returning all
    /// removed characters as an iterator.
    ///
    /// The returned iterator keeps a mutable borrow on the string to optimize
    /// its implementation.
    ///
    /// # Panics
    ///
    /// Panics if the starting point or end point do not lie on a [`char`]
    /// boundary, or if they're out of bounds.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without being dropped (due to
    /// [`core::mem::forget`], for example), the string may still contain a copy
    /// of any drained characters, or may have lost characters arbitrarily,
    /// including characters outside the range.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("α is alpha, β is beta");
    /// let beta_offset = s.find('β').unwrap_or(s.len());
    ///
    /// // Remove the range up until the β from the string
    /// let t: String = s.drain(..beta_offset).collect();
    /// assert_eq!(t, "α is alpha, ");
    /// assert_eq!(s, "β is beta");
    ///
    /// // A full range clears the string, like `clear()` does
    /// s.drain(..);
    /// assert_eq!(s, "");
    /// ```
    #[stable(feature = "drain", since = "1.6.0")]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, A>
    where
        R: RangeBounds<usize>,
    {
        // Memory safety
        //
        // The String version of Drain does not have the memory safety issues
        // of the vector version. The data is just plain bytes.
        // Because the range removal happens in Drop, if the Drain iterator is leaked,
        // the removal will not happen.
        let Range { start, end } = slice::range(range, ..self.len());
        assert!(self.is_char_boundary(start));
        assert!(self.is_char_boundary(end));

        // Take out two simultaneous borrows. The &mut String won't be accessed
        // until iteration is over, in Drop.
        let self_ptr = self as *mut _;
        // SAFETY: `slice::range` and `is_char_boundary` do the appropriate bounds checks.
        let chars_iter = unsafe { self.get_unchecked(start..end) }.chars();

        Drain { start, end, iter: chars_iter, string: self_ptr }
    }

    /// Converts a `String` into an iterator over the [`char`]s of the string.
    ///
    /// As a string consists of valid UTF-8, we can iterate through a string
    /// by [`char`]. This method returns such an iterator.
    ///
    /// It's important to remember that [`char`] represents a Unicode Scalar
    /// Value, and might not match your idea of what a 'character' is. Iteration
    /// over grapheme clusters may be what you actually want. That functionality
    /// is not provided by Rust's standard library, check crates.io instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(string_into_chars)]
    ///
    /// let word = String::from("goodbye");
    ///
    /// let mut chars = word.into_chars();
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
    /// ```
    /// #![feature(string_into_chars)]
    ///
    /// let y = String::from("y̆");
    ///
    /// let mut chars = y.into_chars();
    ///
    /// assert_eq!(Some('y'), chars.next()); // not 'y̆'
    /// assert_eq!(Some('\u{0306}'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    ///
    /// [`char`]: prim@char
    #[inline]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[unstable(feature = "string_into_chars", issue = "133125")]
    pub fn into_chars(self) -> IntoChars<A> {
        IntoChars { bytes: self.into_bytes().into_iter() }
    }


    /// Removes the specified range in the string,
    /// and replaces it with the given string.
    /// The given string doesn't need to be the same length as the range.
    ///
    /// # Panics
    ///
    /// Panics if the starting point or end point do not lie on a [`char`]
    /// boundary, or if they're out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("α is alpha, β is beta");
    /// let beta_offset = s.find('β').unwrap_or(s.len());
    ///
    /// // Replace the range up until the β from the string
    /// s.replace_range(..beta_offset, "Α is capital alpha; ");
    /// assert_eq!(s, "Α is capital alpha; β is beta");
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "splice", since = "1.27.0")]
    pub fn replace_range<R>(&mut self, range: R, replace_with: &str)
    where
        R: RangeBounds<usize>,
    {
        // Memory safety
        //
        // Replace_range does not have the memory safety issues of a vector Splice.
        // of the vector version. The data is just plain bytes.

        // WARNING: Inlining this variable would be unsound (#81138)
        let start = range.start_bound();
        match start {
            Included(&n) => assert!(self.is_char_boundary(n)),
            Excluded(&n) => assert!(self.is_char_boundary(n + 1)),
            Unbounded => {}
        };
        // WARNING: Inlining this variable would be unsound (#81138)
        let end = range.end_bound();
        match end {
            Included(&n) => assert!(self.is_char_boundary(n + 1)),
            Excluded(&n) => assert!(self.is_char_boundary(n)),
            Unbounded => {}
        };

        // Using `range` again would be unsound (#81138)
        // We assume the bounds reported by `range` remain the same, but
        // an adversarial implementation could change between calls
        unsafe { self.as_mut_vec() }.splice((start, end), replace_with.bytes());
    }

    /// Converts this `String` into a <code>[Box]<[str]></code>.
    ///
    /// Before doing the conversion, this method discards excess capacity like [`shrink_to_fit`].
    /// Note that this call may reallocate and copy the bytes of the string.
    ///
    /// [`shrink_to_fit`]: String::shrink_to_fit
    /// [str]: prim@str "str"
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from("hello");
    ///
    /// let b = s.into_boxed_str();
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "box_str", since = "1.4.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub fn into_boxed_str(self) -> Box<str, A> {
        let slice = self.vec.into_boxed_slice();
        unsafe { from_boxed_utf8_unchecked(slice) }
    }

    /// Consumes and leaks the `String`, returning a mutable reference to the contents,
    /// `&'a mut str`.
    ///
    /// The caller has free choice over the returned lifetime, including `'static`. Indeed,
    /// this function is ideally used for data that lives for the remainder of the program's life,
    /// as dropping the returned reference will cause a memory leak.
    ///
    /// It does not reallocate or shrink the `String`, so the leaked allocation may include unused
    /// capacity that is not part of the returned slice. If you want to discard excess capacity,
    /// call [`into_boxed_str`], and then [`Box::leak`] instead. However, keep in mind that
    /// trimming the capacity may result in a reallocation and copy.
    ///
    /// [`into_boxed_str`]: Self::into_boxed_str
    ///
    /// # Examples
    ///
    /// ```
    /// let x = String::from("bucket");
    /// let static_ref: &'static mut str = x.leak();
    /// assert_eq!(static_ref, "bucket");
    /// # // FIXME(https://github.com/rust-lang/miri/issues/3670):
    /// # // use -Zmiri-disable-leak-check instead of unleaking in tests meant to leak.
    /// # drop(unsafe { Box::from_raw(static_ref) });
    /// ```
    #[stable(feature = "string_leak", since = "1.72.0")]
    #[inline]
    pub fn leak<'a>(self) -> &'a mut str
    where
        A: 'a,
    {
        let slice = self.vec.leak();
        unsafe { from_utf8_unchecked_mut(slice) }
    }
}

impl<A: Allocator> FromUtf8Error<A> {
    /// Returns a slice of [`u8`]s bytes that were attempted to convert to a `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// // some invalid bytes, in a vector
    /// let bytes = vec![0, 159];
    ///
    /// let value = String::from_utf8(bytes);
    ///
    /// assert_eq!(&[0, 159], value.unwrap_err().as_bytes());
    /// ```
    #[must_use]
    #[stable(feature = "from_utf8_error_as_bytes", since = "1.26.0")]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..]
    }

    /// Converts the bytes into a `String` lossily, substituting invalid UTF-8
    /// sequences with replacement characters.
    ///
    /// See [`String::from_utf8_lossy`] for more details on replacement of
    /// invalid sequences, and [`String::from_utf8_lossy_owned`] for the
    /// `String` function which corresponds to this function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(string_from_utf8_lossy_owned)]
    /// // some invalid bytes
    /// let input: Vec<u8> = b"Hello \xF0\x90\x80World".into();
    /// let output = String::from_utf8(input).unwrap_or_else(|e| e.into_utf8_lossy());
    ///
    /// assert_eq!(String::from("Hello �World"), output);
    /// ```
    #[must_use]
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "string_from_utf8_lossy_owned", issue = "129436")]
    pub fn into_utf8_lossy(self) -> String {
        const REPLACEMENT: &str = "\u{FFFD}";

        let mut res = {
            let mut v = Vec::with_capacity(self.bytes.len());

            // `Utf8Error::valid_up_to` returns the maximum index of validated
            // UTF-8 bytes. Copy the valid bytes into the output buffer.
            v.extend_from_slice(&self.bytes[..self.error.valid_up_to()]);

            // SAFETY: This is safe because the only bytes present in the buffer
            // were validated as UTF-8 by the call to `String::from_utf8` which
            // produced this `FromUtf8Error`.
            unsafe { String::from_utf8_unchecked(v) }
        };

        let iter = self.bytes[self.error.valid_up_to()..].utf8_chunks();

        for chunk in iter {
            res.push_str(chunk.valid());
            if !chunk.invalid().is_empty() {
                res.push_str(REPLACEMENT);
            }
        }

        res
    }


    /// Returns the bytes that were attempted to convert to a `String`.
    ///
    /// This method is carefully constructed to avoid allocation. It will
    /// consume the error, moving out the bytes, so that a copy of the bytes
    /// does not need to be made.
    ///
    /// # Examples
    ///
    /// ```
    /// // some invalid bytes, in a vector
    /// let bytes = vec![0, 159];
    ///
    /// let value = String::from_utf8(bytes);
    ///
    /// assert_eq!(vec![0, 159], value.unwrap_err().into_bytes());
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_bytes(self) -> Vec<u8, A> {
        self.bytes
    }

    /// Fetch a `Utf8Error` to get more details about the conversion failure.
    ///
    /// The [`Utf8Error`] type provided by [`std::str`] represents an error that may
    /// occur when converting a slice of [`u8`]s to a [`&str`]. In this sense, it's
    /// an analogue to `FromUtf8Error`. See its documentation for more details
    /// on using it.
    ///
    /// [`std::str`]: core::str "std::str"
    /// [`&str`]: prim@str "&str"
    ///
    /// # Examples
    ///
    /// ```
    /// // some invalid bytes, in a vector
    /// let bytes = vec![0, 159];
    ///
    /// let error = String::from_utf8(bytes).unwrap_err().utf8_error();
    ///
    /// // the first byte is invalid here
    /// assert_eq!(1, error.valid_up_to());
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn utf8_error(&self) -> Utf8Error {
        self.error
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> fmt::Display for FromUtf8Error<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for FromUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt("invalid utf-16: lone surrogate found", f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Error for FromUtf8Error<A> {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "invalid utf-8"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for FromUtf16Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "invalid utf-16"
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator + Clone> Clone for String<A> {
    fn clone(&self) -> Self {
        String { vec: self.vec.clone() }
    }

    /// Clones the contents of `source` into `self`.
    ///
    /// This method is preferred over simply assigning `source.clone()` to `self`,
    /// as it avoids reallocation if possible.
    fn clone_from(&mut self, source: &Self) {
        self.vec.clone_from(&source.vec);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl FromIterator<char> for String {
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> String {
        let mut buf = String::new();
        buf.extend(iter);
        buf
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "string_from_iter_by_ref", since = "1.17.0")]
impl<'a> FromIterator<&'a char> for String {
    fn from_iter<I: IntoIterator<Item = &'a char>>(iter: I) -> String {
        let mut buf = String::new();
        buf.extend(iter);
        buf
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> FromIterator<&'a str> for String {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> String {
        let mut buf = String::new();
        buf.extend(iter);
        buf
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "extend_string", since = "1.4.0")]
impl FromIterator<String> for String {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> String {
        let mut iterator = iter.into_iter();

        // Because we're iterating over `String`s, we can avoid at least
        // one allocation by getting the first string from the iterator
        // and appending to it all the subsequent strings.
        match iterator.next() {
            None => String::new(),
            Some(mut buf) => {
                buf.extend(iterator);
                buf
            }
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "box_str2", since = "1.45.0")]
impl<A: Allocator> FromIterator<Box<str, A>> for String {
    fn from_iter<I: IntoIterator<Item = Box<str, A>>>(iter: I) -> String {
        let mut buf = String::new();
        buf.extend(iter);
        buf
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "herd_cows", since = "1.19.0")]
impl<'a> FromIterator<Cow<'a, str>> for String {
    fn from_iter<I: IntoIterator<Item = Cow<'a, str>>>(iter: I) -> String {
        let mut iterator = iter.into_iter();

        // Because we're iterating over CoWs, we can (potentially) avoid at least
        // one allocation by getting the first item and appending to it all the
        // subsequent items.
        match iterator.next() {
            None => String::new(),
            Some(cow) => {
                let mut buf = cow.into_owned();
                buf.extend(iterator);
                buf
            }
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Extend<char> for String<A> {
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let (lower_bound, _) = iterator.size_hint();
        self.reserve(lower_bound);
        iterator.for_each(move |c| self.push(c));
    }

    #[inline]
    fn extend_one(&mut self, c: char) {
        self.push(c);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, A: Allocator> Extend<&'a char> for String<A> {
    fn extend<I: IntoIterator<Item = &'a char>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }

    #[inline]
    fn extend_one(&mut self, &c: &'a char) {
        self.push(c);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A: Allocator> Extend<&'a str> for String<A> {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        iter.into_iter().for_each(move |s| self.push_str(s));
    }

    #[inline]
    fn extend_one(&mut self, s: &'a str) {
        self.push_str(s);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "box_str2", since = "1.45.0")]
impl<A1: Allocator, A2: Allocator> Extend<Box<str, A2>> for String<A1> {
    fn extend<I: IntoIterator<Item = Box<str, A2>>>(&mut self, iter: I) {
        iter.into_iter().for_each(move |s| self.push_str(&s));
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "extend_string", since = "1.4.0")]
impl<A1: Allocator, A2: Allocator> Extend<String<A2>> for String<A1> {
    fn extend<I: IntoIterator<Item = String<A2>>>(&mut self, iter: I) {
        iter.into_iter().for_each(move |s| self.push_str(&s));
    }

    #[inline]
    fn extend_one(&mut self, s: String<A2>) {
        self.push_str(&s);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "herd_cows", since = "1.19.0")]
impl<'a, A: Allocator> Extend<Cow<'a, str>> for String<A> {
    fn extend<I: IntoIterator<Item = Cow<'a, str>>>(&mut self, iter: I) {
        iter.into_iter().for_each(move |s| self.push_str(&s));
    }

    #[inline]
    fn extend_one(&mut self, s: Cow<'a, str>) {
        self.push_str(&s);
    }
}

/// A convenience impl that delegates to the impl for `&str`.
///
/// # Examples
///
/// ```
/// assert_eq!(String::from("Hello world").find("world"), Some(6));
/// ```
#[unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]
impl<'b, A: Allocator> Pattern for &'b String<A> {
    type Searcher<'a> = <&'b str as Pattern>::Searcher<'a>;

    fn into_searcher(self, haystack: &str) -> <&'b str as Pattern>::Searcher<'_> {
        self[..].into_searcher(haystack)
    }

    #[inline]
    fn is_contained_in(self, haystack: &str) -> bool {
        self[..].is_contained_in(haystack)
    }

    #[inline]
    fn is_prefix_of(self, haystack: &str) -> bool {
        self[..].is_prefix_of(haystack)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &str) -> Option<&str> {
        self[..].strip_prefix_of(haystack)
    }

    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a str) -> bool
    where
        Self::Searcher<'a>: core::str::pattern::ReverseSearcher<'a>,
    {
        self[..].is_suffix_of(haystack)
    }

    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a str) -> Option<&'a str>
    where
        Self::Searcher<'a>: core::str::pattern::ReverseSearcher<'a>,
    {
        self[..].strip_suffix_of(haystack)
    }

    #[inline]
    fn as_utf8_pattern(&self) -> Option<Utf8Pattern<'_>> {
        Some(Utf8Pattern::StringPattern(self.as_bytes()))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator, B: Allocator> PartialEq<String<B>> for String<A> {
    #[inline]
    fn eq(&self, other: &String<B>) -> bool {
        PartialEq::eq(&self[..], &other[..])
    }
    #[inline]
    fn ne(&self, other: &String<B>) -> bool {
        PartialEq::ne(&self[..], &other[..])
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Eq for String<A> {} // FIXME(zachs18): Structural(Partial)Eq?

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator, B: Allocator> PartialOrd<String<B>> for String<A> {
    #[inline]
    fn partial_cmp(&self, other: &String<B>) -> Option<Ordering> {
        str::partial_cmp(&self[..], &other[..])
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Ord for String<A> {
    #[inline]
    fn cmp(&self, other: &String<A>) -> Ordering {
        str::cmp(&self[..], &other[..])
    }
}

macro_rules! impl_eq {
    ([$($vars:tt)*] $lhs:ty, $rhs: ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        #[allow(unused_lifetimes)]
        impl<'a, 'b, $($vars)*> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                PartialEq::eq(&self[..], &other[..])
            }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool {
                PartialEq::ne(&self[..], &other[..])
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        #[allow(unused_lifetimes)]
        impl<'a, 'b, $($vars)*> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                PartialEq::eq(&self[..], &other[..])
            }
            #[inline]
            fn ne(&self, other: &$lhs) -> bool {
                PartialEq::ne(&self[..], &other[..])
            }
        }
    };
}

impl_eq! { [A: Allocator] String<A>, str }
impl_eq! { [A: Allocator] String<A>, &'a str }
#[cfg(not(no_global_oom_handling))]
impl_eq! { [] Cow<'a, str>, str }
#[cfg(not(no_global_oom_handling))]
impl_eq! { [] Cow<'a, str>, &'b str }
#[cfg(not(no_global_oom_handling))]
impl_eq! { [A: Allocator] Cow<'a, str>, String<A> }

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for String {
    /// Creates an empty `String`.
    #[inline]
    fn default() -> String {
        String::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> fmt::Display for String<A> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> fmt::Debug for String<A> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> hash::Hash for String<A> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}

/// Implements the `+` operator for concatenating two strings.
///
/// This consumes the `String` on the left-hand side and re-uses its buffer (growing it if
/// necessary). This is done to avoid allocating a new `String` and copying the entire contents on
/// every operation, which would lead to *O*(*n*^2) running time when building an *n*-byte string by
/// repeated concatenation.
///
/// The string on the right-hand side is only borrowed; its contents are copied into the returned
/// `String`.
///
/// # Examples
///
/// Concatenating two `String`s takes the first by value and borrows the second:
///
/// ```
/// let a = String::from("hello");
/// let b = String::from(" world");
/// let c = a + &b;
/// // `a` is moved and can no longer be used here.
/// ```
///
/// If you want to keep using the first `String`, you can clone it and append to the clone instead:
///
/// ```
/// let a = String::from("hello");
/// let b = String::from(" world");
/// let c = a.clone() + &b;
/// // `a` is still valid here.
/// ```
///
/// Concatenating `&str` slices can be done by converting the first to a `String`:
///
/// ```
/// let a = "hello";
/// let b = " world";
/// let c = a.to_string() + b;
/// ```
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Add<&str> for String<A> {
    type Output = String<A>;

    #[inline]
    fn add(mut self, other: &str) -> String<A> {
        self.push_str(other);
        self
    }
}

/// Implements the `+=` operator for appending to a `String`.
///
/// This has the same behavior as the [`push_str`][String::push_str] method.
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "stringaddassign", since = "1.12.0")]
impl<A: Allocator> AddAssign<&str> for String<A> {
    #[inline]
    fn add_assign(&mut self, other: &str) {
        self.push_str(other);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I, A> ops::Index<I> for String<A>
where
    I: slice::SliceIndex<str>,
    A: Allocator,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self.as_str())
    }
}

#[stable(feature = "derefmut_for_string", since = "1.3.0")]
impl<I, A> ops::IndexMut<I> for String<A>
where
    I: slice::SliceIndex<str>,
    A: Allocator,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self.as_mut_str())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> ops::Deref for String<A> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<A: Allocator> ops::DerefPure for String<A> {}

#[stable(feature = "derefmut_for_string", since = "1.3.0")]
impl<A: Allocator> ops::DerefMut for String<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for String {
    type Err = core::convert::Infallible;
    #[inline]
    fn from_str(s: &str) -> Result<String, Self::Err> {
        Ok(String::from(s))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> AsRef<str> for String<A> {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

#[stable(feature = "string_as_mut", since = "1.43.0")]
impl<A: Allocator> AsMut<str> for String<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> AsRef<[u8]> for String<A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl From<&str> for String {
    /// Converts a `&str` into a [`String`].
    ///
    /// The result is allocated on the heap.
    #[inline]
    fn from(s: &str) -> String {
        s.to_owned()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_mut_str_for_string", since = "1.44.0")]
impl From<&mut str> for String {
    /// Converts a `&mut str` into a [`String`].
    ///
    /// The result is allocated on the heap.
    #[inline]
    fn from(s: &mut str) -> String {
        s.to_owned()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_ref_string", since = "1.35.0")]
impl From<&String> for String {
    /// Converts a `&String` into a [`String`].
    ///
    /// This clones `s` and returns the clone.
    #[inline]
    fn from(s: &String) -> String {
        s.clone()
    }
}

// note: test pulls in std, which causes errors here
#[cfg(not(test))]
#[stable(feature = "string_from_box", since = "1.18.0")]
impl<A: Allocator> From<Box<str, A>> for String<A> {
    /// Converts the given boxed `str` slice to a [`String`].
    /// It is notable that the `str` slice is owned.
    ///
    /// # Examples
    ///
    /// ```
    /// let s1: String = String::from("hello world");
    /// let s2: Box<str> = s1.into_boxed_str();
    /// let s3: String = String::from(s2);
    ///
    /// assert_eq!("hello world", s3)
    /// ```
    fn from(s: Box<str, A>) -> String<A> {
        s.into_string()
    }
}

// When compiling in test mode, `Box` is not actually local, so this impl is incoherent
// since `A` is the "main" type.
// To work around this, restrict this impl to `A = Global` under cfg(test).
#[cfg(not(test))]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "box_from_str", since = "1.20.0")]
impl<A: Allocator> From<String<A>> for Box<str, A> {
    /// Converts the given [`String`] to a boxed `str` slice that is owned.
    ///
    /// # Examples
    ///
    /// ```
    /// let s1: String = String::from("hello world");
    /// let s2: Box<str> = Box::from(s1);
    /// let s3: String = String::from(s2);
    ///
    /// assert_eq!("hello world", s3)
    /// ```
    fn from(s: String<A>) -> Box<str, A> {
        s.into_boxed_str()
    }
}

// See above `impl<A: Allocator> From<String<A>> for Box<str, A>`
#[cfg(test)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "box_from_str", since = "1.20.0")]
impl From<String> for Box<str> {
    /// Converts the given [`String`] to a boxed `str` slice that is owned.
    ///
    /// # Examples
    ///
    /// ```
    /// let s1: String = String::from("hello world");
    /// let s2: Box<str> = Box::from(s1);
    /// let s3: String = String::from(s2);
    ///
    /// assert_eq!("hello world", s3)
    /// ```
    fn from(s: String) -> Box<str> {
        s.into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "string_from_cow_str", since = "1.14.0")]
impl<'a> From<Cow<'a, str>> for String {
    /// Converts a clone-on-write string to an owned
    /// instance of [`String`].
    ///
    /// This extracts the owned string,
    /// clones the string if it is not already owned.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::borrow::Cow;
    /// // If the string is not owned...
    /// let cow: Cow<'_, str> = Cow::Borrowed("eggplant");
    /// // It will allocate on the heap and copy the string.
    /// let owned: String = String::from(cow);
    /// assert_eq!(&owned[..], "eggplant");
    /// ```
    fn from(s: Cow<'a, str>) -> String {
        s.into_owned()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<&'a str> for Cow<'a, str> {
    /// Converts a string slice into a [`Borrowed`] variant.
    /// No heap allocation is performed, and the string
    /// is not copied.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::borrow::Cow;
    /// assert_eq!(Cow::from("eggplant"), Cow::Borrowed("eggplant"));
    /// ```
    ///
    /// [`Borrowed`]: crate::borrow::Cow::Borrowed "borrow::Cow::Borrowed"
    #[inline]
    fn from(s: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(s)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<String> for Cow<'a, str> {
    /// Converts a [`String`] into an [`Owned`] variant.
    /// No heap allocation is performed, and the string
    /// is not copied.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::borrow::Cow;
    /// let s = "eggplant".to_string();
    /// let s2 = "eggplant".to_string();
    /// assert_eq!(Cow::from(s), Cow::<'static, str>::Owned(s2));
    /// ```
    ///
    /// [`Owned`]: crate::borrow::Cow::Owned "borrow::Cow::Owned"
    #[inline]
    fn from(s: String) -> Cow<'a, str> {
        Cow::Owned(s)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "cow_from_string_ref", since = "1.28.0")]
impl<'a, A: Allocator> From<&'a String<A>> for Cow<'a, str> {
    /// Converts a [`String`] reference into a [`Borrowed`] variant.
    /// No heap allocation is performed, and the string
    /// is not copied.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::borrow::Cow;
    /// let s = "eggplant".to_string();
    /// assert_eq!(Cow::from(&s), Cow::Borrowed("eggplant"));
    /// ```
    ///
    /// [`Borrowed`]: crate::borrow::Cow::Borrowed "borrow::Cow::Borrowed"
    #[inline]
    fn from(s: &'a String<A>) -> Cow<'a, str> {
        Cow::Borrowed(s.as_str())
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "cow_str_from_iter", since = "1.12.0")]
impl<'a> FromIterator<char> for Cow<'a, str> {
    fn from_iter<I: IntoIterator<Item = char>>(it: I) -> Cow<'a, str> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "cow_str_from_iter", since = "1.12.0")]
impl<'a, 'b> FromIterator<&'b str> for Cow<'a, str> {
    fn from_iter<I: IntoIterator<Item = &'b str>>(it: I) -> Cow<'a, str> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "cow_str_from_iter", since = "1.12.0")]
impl<'a> FromIterator<String> for Cow<'a, str> {
    fn from_iter<I: IntoIterator<Item = String>>(it: I) -> Cow<'a, str> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}

#[stable(feature = "from_string_for_vec_u8", since = "1.14.0")]
impl<A: Allocator> From<String<A>> for Vec<u8, A> {
    /// Converts the given [`String`] to a vector [`Vec`] that holds values of type [`u8`].
    ///
    /// # Examples
    ///
    /// ```
    /// let s1 = String::from("hello world");
    /// let v1 = Vec::from(s1);
    ///
    /// for b in v1 {
    ///     println!("{b}");
    /// }
    /// ```
    fn from(string: String<A>) -> Vec<u8, A> {
        string.into_bytes()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> fmt::Write for String<A> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push(c);
        Ok(())
    }
}

/// An iterator over the [`char`]s of a string.
///
/// This struct is created by the [`into_chars`] method on [`String`].
/// See its documentation for more.
///
/// [`char`]: prim@char
/// [`into_chars`]: String::into_chars
#[cfg_attr(not(no_global_oom_handling), derive(Clone))]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "string_into_chars", issue = "133125")]
pub struct IntoChars<A: Allocator> {
    bytes: vec::IntoIter<u8, A>,
}

#[unstable(feature = "string_into_chars", issue = "133125")]
impl<A: Allocator> fmt::Debug for IntoChars<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoChars").field(&self.as_str()).finish()
    }
}

impl<A: Allocator> IntoChars<A> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(string_into_chars)]
    ///
    /// let mut chars = String::from("abc").into_chars();
    ///
    /// assert_eq!(chars.as_str(), "abc");
    /// chars.next();
    /// assert_eq!(chars.as_str(), "bc");
    /// chars.next();
    /// chars.next();
    /// assert_eq!(chars.as_str(), "");
    /// ```
    #[unstable(feature = "string_into_chars", issue = "133125")]
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `bytes` is a valid UTF-8 string.
        unsafe { str::from_utf8_unchecked(self.bytes.as_slice()) }
    }

    #[inline]
    fn iter(&self) -> CharIndices<'_> {
        self.as_str().char_indices()
    }
}

impl IntoChars<Global> {
    /// Consumes the `IntoChars`, returning the remaining string.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(string_into_chars)]
    ///
    /// let chars = String::from("abc").into_chars();
    /// assert_eq!(chars.into_string(), "abc");
    ///
    /// let mut chars = String::from("def").into_chars();
    /// chars.next();
    /// assert_eq!(chars.into_string(), "ef");
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "string_into_chars", issue = "133125")]
    #[inline]
    pub fn into_string(self) -> String {
        // Safety: `bytes` are kept in UTF-8 form, only removing whole `char`s at a time.
        unsafe { String::from_utf8_unchecked(self.bytes.collect()) }
    }
}

#[unstable(feature = "string_into_chars", issue = "133125")]
impl<A: Allocator> Iterator for IntoChars<A> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        let mut iter = self.iter();
        match iter.next() {
            None => None,
            Some((_, ch)) => {
                let offset = iter.offset();
                // `offset` is a valid index.
                let _ = self.bytes.advance_by(offset);
                Some(ch)
            }
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.iter().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        self.next_back()
    }
}

#[unstable(feature = "string_into_chars", issue = "133125")]
impl<A: Allocator> DoubleEndedIterator for IntoChars<A> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        let len = self.as_str().len();
        let mut iter = self.iter();
        match iter.next_back() {
            None => None,
            Some((idx, ch)) => {
                // `idx` is a valid index.
                let _ = self.bytes.advance_back_by(len - idx);
                Some(ch)
            }
        }
    }
}

#[unstable(feature = "string_into_chars", issue = "133125")]
impl<A: Allocator> FusedIterator for IntoChars<A> {}


/// A draining iterator for `String`.
///
/// This struct is created by the [`drain`] method on [`String`]. See its
/// documentation for more.
///
/// [`drain`]: String::drain
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<'a, #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global>
{
    /// Will be used as &'a mut String in the destructor
    string: *mut String<A>,
    /// Start of part to remove
    start: usize,
    /// End of part to remove
    end: usize,
    /// Current remaining range to remove
    iter: Chars<'a>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<A: Allocator> fmt::Debug for Drain<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain").field(&self.as_str()).finish()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<A: Allocator + Sync> Sync for Drain<'_, A> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<A: Allocator + Send> Send for Drain<'_, A> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<A: Allocator> Drop for Drain<'_, A> {
    fn drop(&mut self) {
        unsafe {
            // Use Vec::drain. "Reaffirm" the bounds checks to avoid
            // panic code being inserted again.
            let self_vec = (*self.string).as_mut_vec();
            if self.start <= self.end && self.end <= self_vec.len() {
                self_vec.drain(self.start..self.end);
            }
        }
    }
}

impl<'a, A: Allocator> Drain<'a, A> {
    /// Returns the remaining (sub)string of this iterator as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("abc");
    /// let mut drain = s.drain(..);
    /// assert_eq!(drain.as_str(), "abc");
    /// let _ = drain.next().unwrap();
    /// assert_eq!(drain.as_str(), "bc");
    /// ```
    #[must_use]
    #[stable(feature = "string_drain_as_str", since = "1.55.0")]
    pub fn as_str(&self) -> &str {
        self.iter.as_str()
    }
}

#[stable(feature = "string_drain_as_str", since = "1.55.0")]
impl<'a, A: Allocator> AsRef<str> for Drain<'a, A> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[stable(feature = "string_drain_as_str", since = "1.55.0")]
impl<'a, A: Allocator> AsRef<[u8]> for Drain<'a, A> {
    fn as_ref(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<A: Allocator> Iterator for Drain<'_, A> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        self.next_back()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<A: Allocator> DoubleEndedIterator for Drain<'_, A> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        self.iter.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Allocator> FusedIterator for Drain<'_, A> {}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_char_for_string", since = "1.46.0")]
impl From<char> for String {
    /// Allocates an owned [`String`] from a single character.
    ///
    /// # Example
    /// ```rust
    /// let c: char = 'a';
    /// let s: String = String::from(c);
    /// assert_eq!("a", &s[..]);
    /// ```
    #[inline]
    fn from(c: char) -> Self {
        c.to_string()
    }
}
