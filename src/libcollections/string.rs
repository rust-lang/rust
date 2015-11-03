// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An owned, growable string that enforces that its contents are valid UTF-8.

#![stable(feature = "rust1", since = "1.0.0")]

use core::fmt;
use core::hash;
use core::iter::FromIterator;
use core::mem;
use core::ops::{self, Deref, Add, Index};
use core::ptr;
use core::slice;
use core::str::pattern::Pattern;
use rustc_unicode::char::{decode_utf16, REPLACEMENT_CHARACTER};
use rustc_unicode::str as unicode_str;

use borrow::{Cow, IntoCow};
use range::RangeArgument;
use str::{self, FromStr, Utf8Error, Chars};
use vec::Vec;
use boxed::Box;

/// A growable string stored as a UTF-8 encoded buffer.
#[derive(PartialOrd, Eq, Ord)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct String {
    vec: Vec<u8>,
}

/// A possible error value from the `String::from_utf8` function.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct FromUtf8Error {
    bytes: Vec<u8>,
    error: Utf8Error,
}

/// A possible error value from the `String::from_utf16` function.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct FromUtf16Error(());

impl String {
    /// Creates a new string buffer initialized with the empty string.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// let mut s = String::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> String {
        String {
            vec: Vec::new(),
        }
    }

    /// Creates a new string buffer with the given capacity.
    /// The string will be able to hold exactly `capacity` bytes without
    /// reallocating. If `capacity` is 0, the string will not allocate.
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
    /// for i in 0..10 {
    ///     s.push('a');
    /// }
    ///
    /// assert_eq!(s.capacity(), cap);
    ///
    /// // ...but this may make the vector reallocate
    /// s.push('a');
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> String {
        String {
            vec: Vec::with_capacity(capacity),
        }
    }

    // HACK(japaric): with cfg(test) the inherent `[T]::to_vec` method, which is
    // required for this method definition, is not available. Since we don't
    // require this method for testing purposes, I'll just stub it
    // NB see the slice::hack module in slice.rs for more information
    #[inline]
    #[cfg(test)]
    pub fn from_str(_: &str) -> String {
        panic!("not available with cfg(test)");
    }

    /// Converts a vector of bytes to a `String`.
    ///
    /// A string slice (`&str`) is made of bytes (`u8`), and a vector of bytes
    /// (`Vec<u8>`) is made of bytes, so this function converts between the
    /// two. Not all byte slices are valid `String`s, however: `String`
    /// requires that it is valid UTF-8. `from_utf8()` checks to ensure that
    /// the bytes are valid UTF-8, and then does the conversion.
    ///
    /// If you are sure that the byte slice is valid UTF-8, and you don't want
    /// to incur the overhead of the validity check, there is an unsafe version
    /// of this function, [`from_utf8_unchecked()`][fromutf8], which has the
    /// same behavior but skips the check.
    ///
    /// [fromutf8]: struct.String.html#method.from_utf8_unchecked
    ///
    /// This method will take care to not copy the vector, for efficiency's
    /// sake.
    ///
    /// If you need a `&str` instead of a `String`, consider
    /// [`str::from_utf8()`][str].
    ///
    /// [str]: ../str/fn.from_utf8.html
    ///
    /// # Failure
    ///
    /// Returns `Err` if the slice is not UTF-8 with a description as to why the
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
    /// // We know these bytes are valid, so just use `unwrap()`.
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
    /// See the docs for [`FromUtf8Error`][error] for more details on what you
    /// can do with this error.
    ///
    /// [error]: struct.FromUtf8Error.html
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf8(vec: Vec<u8>) -> Result<String, FromUtf8Error> {
        match str::from_utf8(&vec) {
            Ok(..) => Ok(String { vec: vec }),
            Err(e) => Err(FromUtf8Error { bytes: vec, error: e })
        }
    }

    /// Converts a slice of bytes to a `String`, including invalid characters.
    ///
    /// A string slice (`&str`) is made of bytes (`u8`), and a slice of bytes
    /// (`&[u8]`) is made of bytes, so this function converts between the two.
    /// Not all byte slices are valid string slices, however: `&str` requires
    /// that it is valid UTF-8. During this conversion, `from_utf8_lossy()`
    /// will replace any invalid UTF-8 sequences with
    /// `U+FFFD REPLACEMENT CHARACTER`, which looks like this: �
    ///
    /// If you are sure that the byte slice is valid UTF-8, and you don't want
    /// to incur the overhead of the conversion, there is an unsafe version
    /// of this function, [`from_utf8_unchecked()`][fromutf8], which has the
    /// same behavior but skips the checks.
    ///
    /// [fromutf8]: struct.String.html#method.from_utf8_unchecked
    ///
    /// If you need a `&str` instead of a `String`, consider
    /// [`str::from_utf8()`][str].
    ///
    /// [str]: ../str/fn.from_utf8.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // some bytes, in a vector
    /// let sparkle_heart = vec![240, 159, 146, 150];
    ///
    /// // We know these bytes are valid, so just use `unwrap()`.
    /// let sparkle_heart = String::from_utf8(sparkle_heart).unwrap();
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf8_lossy<'a>(v: &'a [u8]) -> Cow<'a, str> {
        let mut i;
        match str::from_utf8(v) {
            Ok(s) => return Cow::Borrowed(s),
            Err(e) => i = e.valid_up_to(),
        }

        const TAG_CONT_U8: u8 = 128;
        const REPLACEMENT: &'static [u8] = b"\xEF\xBF\xBD"; // U+FFFD in UTF-8
        let total = v.len();
        fn unsafe_get(xs: &[u8], i: usize) -> u8 {
            unsafe { *xs.get_unchecked(i) }
        }
        fn safe_get(xs: &[u8], i: usize, total: usize) -> u8 {
            if i >= total {
                0
            } else {
                unsafe_get(xs, i)
            }
        }

        let mut res = String::with_capacity(total);

        if i > 0 {
            unsafe {
                res.as_mut_vec().push_all(&v[..i])
            };
        }

        // subseqidx is the index of the first byte of the subsequence we're
        // looking at.  It's used to copy a bunch of contiguous good codepoints
        // at once instead of copying them one by one.
        let mut subseqidx = i;

        while i < total {
            let i_ = i;
            let byte = unsafe_get(v, i);
            i += 1;

            macro_rules! error { () => ({
                unsafe {
                    if subseqidx != i_ {
                        res.as_mut_vec().push_all(&v[subseqidx..i_]);
                    }
                    subseqidx = i;
                    res.as_mut_vec().push_all(REPLACEMENT);
                }
            })}

            if byte < 128 {
                // subseqidx handles this
            } else {
                let w = unicode_str::utf8_char_width(byte);

                match w {
                    2 => {
                        if safe_get(v, i, total) & 192 != TAG_CONT_U8 {
                            error!();
                            continue;
                        }
                        i += 1;
                    }
                    3 => {
                        match (byte, safe_get(v, i, total)) {
                            (0xE0         , 0xA0 ... 0xBF) => (),
                            (0xE1 ... 0xEC, 0x80 ... 0xBF) => (),
                            (0xED         , 0x80 ... 0x9F) => (),
                            (0xEE ... 0xEF, 0x80 ... 0xBF) => (),
                            _ => {
                                error!();
                                continue;
                            }
                        }
                        i += 1;
                        if safe_get(v, i, total) & 192 != TAG_CONT_U8 {
                            error!();
                            continue;
                        }
                        i += 1;
                    }
                    4 => {
                        match (byte, safe_get(v, i, total)) {
                            (0xF0         , 0x90 ... 0xBF) => (),
                            (0xF1 ... 0xF3, 0x80 ... 0xBF) => (),
                            (0xF4         , 0x80 ... 0x8F) => (),
                            _ => {
                                error!();
                                continue;
                            }
                        }
                        i += 1;
                        if safe_get(v, i, total) & 192 != TAG_CONT_U8 {
                            error!();
                            continue;
                        }
                        i += 1;
                        if safe_get(v, i, total) & 192 != TAG_CONT_U8 {
                            error!();
                            continue;
                        }
                        i += 1;
                    }
                    _ => {
                        error!();
                        continue;
                    }
                }
            }
        }
        if subseqidx < total {
            unsafe {
                res.as_mut_vec().push_all(&v[subseqidx..total])
            };
        }
        Cow::Owned(res)
    }

    /// Decode a UTF-16 encoded vector `v` into a `String`, returning `None`
    /// if `v` contains any invalid data.
    ///
    /// # Examples
    ///
    /// ```
    /// // 𝄞music
    /// let mut v = &mut [0xD834, 0xDD1E, 0x006d, 0x0075,
    ///                   0x0073, 0x0069, 0x0063];
    /// assert_eq!(String::from_utf16(v).unwrap(),
    ///            "𝄞music".to_string());
    ///
    /// // 𝄞mu<invalid>ic
    /// v[4] = 0xD800;
    /// assert!(String::from_utf16(v).is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf16(v: &[u16]) -> Result<String, FromUtf16Error> {
        decode_utf16(v.iter().cloned()).collect::<Result<_, _>>().map_err(|_| FromUtf16Error(()))
    }

    /// Decode a UTF-16 encoded vector `v` into a string, replacing
    /// invalid data with the replacement character (U+FFFD).
    ///
    /// # Examples
    ///
    /// ```
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075,
    ///           0x0073, 0xDD1E, 0x0069, 0x0063,
    ///           0xD834];
    ///
    /// assert_eq!(String::from_utf16_lossy(v),
    ///            "𝄞mus\u{FFFD}ic\u{FFFD}".to_string());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf16_lossy(v: &[u16]) -> String {
        decode_utf16(v.iter().cloned()).map(|r| r.unwrap_or(REPLACEMENT_CHARACTER)).collect()
    }

    /// Creates a new `String` from a length, capacity, and pointer.
    ///
    /// # Safety
    ///
    /// This is _very_ unsafe because:
    ///
    /// * We call `Vec::from_raw_parts` to get a `Vec<u8>`. Therefore, this
    ///   function inherits all of its unsafety, see [its
    ///   documentation](../vec/struct.Vec.html#method.from_raw_parts)
    ///   for the invariants it expects, they also apply to this function.
    /// * We assume that the `Vec` contains valid UTF-8.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_raw_parts(buf: *mut u8, length: usize, capacity: usize) -> String {
        String {
            vec: Vec::from_raw_parts(buf, length, capacity),
        }
    }

    /// Converts a vector of bytes to a `String` without checking that the
    /// string contains valid UTF-8.
    ///
    /// See the safe version, [`from_utf8()`][fromutf8], for more.
    ///
    /// [fromutf8]: struct.String.html#method.from_utf8
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check that the bytes passed to
    /// it are valid UTF-8. If this constraint is violated, undefined behavior
    /// results, as the rest of Rust assumes that `String`s are valid UTF-8.
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
    ///     String::from_utf8_unchecked(sparkle_heart)
    /// };
    ///
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_utf8_unchecked(bytes: Vec<u8>) -> String {
        String { vec: bytes }
    }

    /// Returns the underlying byte buffer, encoded as UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from("hello");
    /// let bytes = s.into_bytes();
    /// assert_eq!(bytes, [104, 101, 108, 108, 111]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_bytes(self) -> Vec<u8> {
        self.vec
    }

    /// Extracts a string slice containing the entire string.
    #[inline]
    #[unstable(feature = "convert",
               reason = "waiting on RFC revision",
               issue = "27729")]
    pub fn as_str(&self) -> &str {
        self
    }

    /// Pushes the given string onto this string buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    /// s.push_str("bar");
    /// assert_eq!(s, "foobar");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push_str(&mut self, string: &str) {
        self.vec.push_all(string.as_bytes())
    }

    /// Returns the number of bytes that this string buffer can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::with_capacity(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    /// Reserves capacity for at least `additional` more bytes to be inserted
    /// in the given `String`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::new();
    /// s.reserve(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        self.vec.reserve(additional)
    }

    /// Reserves the minimum capacity for exactly `additional` more bytes to be
    /// inserted in the given `String`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore capacity can not be relied upon to be precisely
    /// minimal. Prefer `reserve` if future insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::new();
    /// s.reserve_exact(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.vec.reserve_exact(additional)
    }

    /// Shrinks the capacity of this string buffer to match its length.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    /// s.reserve(100);
    /// assert!(s.capacity() >= 100);
    /// s.shrink_to_fit();
    /// assert_eq!(s.capacity(), 3);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit()
    }

    /// Adds the given character to the end of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("abc");
    /// s.push('1');
    /// s.push('2');
    /// s.push('3');
    /// assert_eq!(s, "abc123");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, ch: char) {
        match ch.len_utf8() {
            1 => self.vec.push(ch as u8),
            ch_len => {
                let cur_len = self.len();
                // This may use up to 4 bytes.
                self.vec.reserve(ch_len);

                unsafe {
                    // Attempt to not use an intermediate buffer by just pushing bytes
                    // directly onto this string.
                    let slice = slice::from_raw_parts_mut (
                        self.vec.as_mut_ptr().offset(cur_len as isize),
                        ch_len
                    );
                    let used = ch.encode_utf8(slice).unwrap_or(0);
                    self.vec.set_len(cur_len + used);
                }
            }
        }
    }

    /// Works with the underlying buffer as a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from("hello");
    /// assert_eq!(s.as_bytes(), [104, 101, 108, 108, 111]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes(&self) -> &[u8] {
        &self.vec
    }

    /// Shortens a string to the specified length.
    ///
    /// # Panics
    ///
    /// Panics if `new_len` > current length,
    /// or if `new_len` is not a character boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("hello");
    /// s.truncate(2);
    /// assert_eq!(s, "he");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(self.is_char_boundary(new_len));
        self.vec.truncate(new_len)
    }

    /// Removes the last character from the string buffer and returns it.
    /// Returns `None` if this string buffer is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    /// assert_eq!(s.pop(), Some('o'));
    /// assert_eq!(s.pop(), Some('o'));
    /// assert_eq!(s.pop(), Some('f'));
    /// assert_eq!(s.pop(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> Option<char> {
        let len = self.len();
        if len == 0 {
            return None
        }

        let ch = self.char_at_reverse(len);
        unsafe {
            self.vec.set_len(len - ch.len_utf8());
        }
        Some(ch)
    }

    /// Removes the character from the string buffer at byte position `idx` and
    /// returns it.
    ///
    /// # Warning
    ///
    /// This is an O(n) operation as it requires copying every element in the
    /// buffer.
    ///
    /// # Panics
    ///
    /// If `idx` does not lie on a character boundary, or if it is out of
    /// bounds, then this function will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("foo");
    /// assert_eq!(s.remove(0), 'f');
    /// assert_eq!(s.remove(1), 'o');
    /// assert_eq!(s.remove(0), 'o');
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, idx: usize) -> char {
        let len = self.len();
        assert!(idx <= len);

        let ch = self.char_at(idx);
        let next = idx + ch.len_utf8();
        unsafe {
            ptr::copy(self.vec.as_ptr().offset(next as isize),
                      self.vec.as_mut_ptr().offset(idx as isize),
                      len - next);
            self.vec.set_len(len - (next - idx));
        }
        ch
    }

    /// Inserts a character into the string buffer at byte position `idx`.
    ///
    /// # Warning
    ///
    /// This is an O(n) operation as it requires copying every element in the
    /// buffer.
    ///
    /// # Panics
    ///
    /// If `idx` does not lie on a character boundary or is out of bounds, then
    /// this function will panic.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, idx: usize, ch: char) {
        let len = self.len();
        assert!(idx <= len);
        assert!(self.is_char_boundary(idx));
        self.vec.reserve(4);
        let mut bits = [0; 4];
        let amt = ch.encode_utf8(&mut bits).unwrap();

        unsafe {
            ptr::copy(self.vec.as_ptr().offset(idx as isize),
                      self.vec.as_mut_ptr().offset((idx + amt) as isize),
                      len - idx);
            ptr::copy(bits.as_ptr(),
                      self.vec.as_mut_ptr().offset(idx as isize),
                      amt);
            self.vec.set_len(len + amt);
        }
    }

    /// Views the string buffer as a mutable sequence of bytes.
    ///
    /// This is unsafe because it does not check
    /// to ensure that the resulting string will be valid UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from("hello");
    /// unsafe {
    ///     let vec = s.as_mut_vec();
    ///     assert!(vec == &[104, 101, 108, 108, 111]);
    ///     vec.reverse();
    /// }
    /// assert_eq!(s, "olleh");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        &mut self.vec
    }

    /// Returns the number of bytes in this string.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = "foo".to_string();
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.vec.len() }

    /// Returns true if the string contains no bytes
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::new();
    /// assert!(v.is_empty());
    /// v.push('a');
    /// assert!(!v.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Truncates the string, returning it to 0 length.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = "foo".to_string();
    /// s.clear();
    /// assert!(s.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.vec.clear()
    }

    /// Create a draining iterator that removes the specified range in the string
    /// and yields the removed chars from start to end. The element range is
    /// removed even if the iterator is not consumed until the end.
    ///
    /// # Panics
    ///
    /// Panics if the starting point or end point are not on character boundaries,
    /// or if they are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(drain)]
    ///
    /// let mut s = String::from("α is alpha, β is beta");
    /// let beta_offset = s.find('β').unwrap_or(s.len());
    ///
    /// // Remove the range up until the β from the string
    /// let t: String = s.drain(..beta_offset).collect();
    /// assert_eq!(t, "α is alpha, ");
    /// assert_eq!(s, "β is beta");
    ///
    /// // A full range clears the string
    /// s.drain(..);
    /// assert_eq!(s, "");
    /// ```
    #[unstable(feature = "drain",
               reason = "recently added, matches RFC",
               issue = "27711")]
    pub fn drain<R>(&mut self, range: R) -> Drain where R: RangeArgument<usize> {
        // Memory safety
        //
        // The String version of Drain does not have the memory safety issues
        // of the vector version. The data is just plain bytes.
        // Because the range removal happens in Drop, if the Drain iterator is leaked,
        // the removal will not happen.
        let len = self.len();
        let start = *range.start().unwrap_or(&0);
        let end = *range.end().unwrap_or(&len);

        // Take out two simultaneous borrows. The &mut String won't be accessed
        // until iteration is over, in Drop.
        let self_ptr = self as *mut _;
        // slicing does the appropriate bounds checks
        let chars_iter = self[start..end].chars();

        Drain {
            start: start,
            end: end,
            iter: chars_iter,
            string: self_ptr,
        }
    }

    /// Converts the string into `Box<str>`.
    ///
    /// Note that this will drop any excess capacity.
    #[stable(feature = "box_str", since = "1.4.0")]
    pub fn into_boxed_str(self) -> Box<str> {
        let slice = self.vec.into_boxed_slice();
        unsafe { mem::transmute::<Box<[u8]>, Box<str>>(slice) }
    }

    /// Converts the string into `Box<str>`.
    ///
    /// Note that this will drop any excess capacity.
    #[unstable(feature = "box_str2",
               reason = "recently added, matches RFC",
               issue = "27785")]
    #[deprecated(since = "1.4.0", reason = "renamed to `into_boxed_str`")]
    pub fn into_boxed_slice(self) -> Box<str> {
        self.into_boxed_str()
    }
}

impl FromUtf8Error {
    /// Consumes this error, returning the bytes that were attempted to make a
    /// `String` with.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_bytes(self) -> Vec<u8> { self.bytes }

    /// Access the underlying UTF8-error that was the cause of this error.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn utf8_error(&self) -> Utf8Error { self.error }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for FromUtf8Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for FromUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt("invalid utf-16: lone surrogate found", f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for String {
    fn clone(&self) -> Self {
        String { vec: self.vec.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.vec.clone_from(&source.vec);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromIterator<char> for String {
    fn from_iter<I: IntoIterator<Item=char>>(iterable: I) -> String {
        let mut buf = String::new();
        buf.extend(iterable);
        buf
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> FromIterator<&'a str> for String {
    fn from_iter<I: IntoIterator<Item=&'a str>>(iterable: I) -> String {
        let mut buf = String::new();
        buf.extend(iterable);
        buf
    }
}

#[stable(feature = "extend_string", since = "1.4.0")]
impl FromIterator<String> for String {
    fn from_iter<I: IntoIterator<Item=String>>(iterable: I) -> String {
        let mut buf = String::new();
        buf.extend(iterable);
        buf
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Extend<char> for String {
    fn extend<I: IntoIterator<Item=char>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        let (lower_bound, _) = iterator.size_hint();
        self.reserve(lower_bound);
        for ch in iterator {
            self.push(ch)
        }
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a> Extend<&'a char> for String {
    fn extend<I: IntoIterator<Item=&'a char>>(&mut self, iterable: I) {
        self.extend(iterable.into_iter().cloned());
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Extend<&'a str> for String {
    fn extend<I: IntoIterator<Item=&'a str>>(&mut self, iterable: I) {
        for s in iterable {
            self.push_str(s)
        }
    }
}

#[stable(feature = "extend_string", since = "1.4.0")]
impl Extend<String> for String {
    fn extend<I: IntoIterator<Item=String>>(&mut self, iterable: I) {
        for s in iterable {
            self.push_str(&s)
        }
    }
}

/// A convenience impl that delegates to the impl for `&str`
impl<'a, 'b> Pattern<'a> for &'b String {
    type Searcher = <&'b str as Pattern<'a>>::Searcher;

    fn into_searcher(self, haystack: &'a str) -> <&'b str as Pattern<'a>>::Searcher {
        self[..].into_searcher(haystack)
    }

    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        self[..].is_contained_in(haystack)
    }

    #[inline]
    fn is_prefix_of(self, haystack: &'a str) -> bool {
        self[..].is_prefix_of(haystack)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for String {
    #[inline]
    fn eq(&self, other: &String) -> bool { PartialEq::eq(&self[..], &other[..]) }
    #[inline]
    fn ne(&self, other: &String) -> bool { PartialEq::ne(&self[..], &other[..]) }
}

macro_rules! impl_eq {
    ($lhs:ty, $rhs: ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { PartialEq::eq(&self[..], &other[..]) }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { PartialEq::ne(&self[..], &other[..]) }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool { PartialEq::eq(&self[..], &other[..]) }
            #[inline]
            fn ne(&self, other: &$lhs) -> bool { PartialEq::ne(&self[..], &other[..]) }
        }

    }
}

impl_eq! { String, str }
impl_eq! { String, &'a str }
impl_eq! { Cow<'a, str>, str }
impl_eq! { Cow<'a, str>, &'b str }
impl_eq! { Cow<'a, str>, String }

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for String {
    #[inline]
    fn default() -> String {
        String::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for String {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for String {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for String {
    #[inline]
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Add<&'a str> for String {
    type Output = String;

    #[inline]
    fn add(mut self, other: &str) -> String {
        self.push_str(other);
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::Range<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &str {
        &self[..][index]
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeTo<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &str {
        &self[..][index]
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeFrom<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &str {
        &self[..][index]
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeFull> for String {
    type Output = str;

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &str {
        unsafe { str::from_utf8_unchecked(&self.vec) }
    }
}

#[stable(feature = "derefmut_for_string", since = "1.2.0")]
impl ops::IndexMut<ops::Range<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut str {
        &mut self[..][index]
    }
}
#[stable(feature = "derefmut_for_string", since = "1.2.0")]
impl ops::IndexMut<ops::RangeTo<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut str {
        &mut self[..][index]
    }
}
#[stable(feature = "derefmut_for_string", since = "1.2.0")]
impl ops::IndexMut<ops::RangeFrom<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut str {
        &mut self[..][index]
    }
}
#[stable(feature = "derefmut_for_string", since = "1.2.0")]
impl ops::IndexMut<ops::RangeFull> for String {
    #[inline]
    fn index_mut(&mut self, _index: ops::RangeFull) -> &mut str {
        unsafe { mem::transmute(&mut *self.vec) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for String {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        unsafe { str::from_utf8_unchecked(&self.vec) }
    }
}

#[stable(feature = "derefmut_for_string", since = "1.2.0")]
impl ops::DerefMut for String {
    #[inline]
    fn deref_mut(&mut self) -> &mut str {
        unsafe { mem::transmute(&mut *self.vec) }
    }
}

/// Error returned from `String::from`
#[stable(feature = "str_parse_error", since = "1.5.0")]
#[derive(Copy)]
pub enum ParseError {}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for String {
    type Err = ParseError;
    #[inline]
    fn from_str(s: &str) -> Result<String, ParseError> {
        Ok(String::from(s))
    }
}

impl Clone for ParseError {
    fn clone(&self) -> ParseError {
        match *self {}
    }
}

impl fmt::Debug for ParseError {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        match *self {}
    }
}

impl PartialEq for ParseError {
    fn eq(&self, _: &ParseError) -> bool {
        match *self {}
    }
}

impl Eq for ParseError {}

/// A generic trait for converting a value to a string
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ToString {
    /// Converts the value of `self` to an owned string
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_string(&self) -> String;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Display + ?Sized> ToString for T {
    #[inline]
    fn to_string(&self) -> String {
        use core::fmt::Write;
        let mut buf = String::new();
        let _ = buf.write_fmt(format_args!("{}", self));
        buf.shrink_to_fit();
        buf
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<str> for String {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<[u8]> for String {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<&'a str> for String {
    #[cfg(not(test))]
    #[inline]
    fn from(s: &'a str) -> String {
        String { vec: <[_]>::to_vec(s.as_bytes()) }
    }

    // HACK(japaric): with cfg(test) the inherent `[T]::to_vec` method, which is
    // required for this method definition, is not available. Since we don't
    // require this method for testing purposes, I'll just stub it
    // NB see the slice::hack module in slice.rs for more information
    #[inline]
    #[cfg(test)]
    fn from(_: &str) -> String {
        panic!("not available with cfg(test)");
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<&'a str> for Cow<'a, str> {
    #[inline]
    fn from(s: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<String> for Cow<'a, str> {
    #[inline]
    fn from(s: String) -> Cow<'a, str> {
        Cow::Owned(s)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Into<Vec<u8>> for String {
    fn into(self) -> Vec<u8> {
        self.into_bytes()
    }
}

#[unstable(feature = "into_cow", reason = "may be replaced by `convert::Into`",
           issue= "27735")]
impl IntoCow<'static, str> for String {
    #[inline]
    fn into_cow(self) -> Cow<'static, str> {
        Cow::Owned(self)
    }
}

#[unstable(feature = "into_cow", reason = "may be replaced by `convert::Into`",
           issue = "27735")]
impl<'a> IntoCow<'a, str> for &'a str {
    #[inline]
    fn into_cow(self) -> Cow<'a, str> {
        Cow::Borrowed(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Write for String {
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

/// A draining iterator for `String`.
#[unstable(feature = "drain", reason = "recently added", issue = "27711")]
pub struct Drain<'a> {
    /// Will be used as &'a mut String in the destructor
    string: *mut String,
    /// Start of part to remove
    start: usize,
    /// End of part to remove
    end: usize,
    /// Current remaining range to remove
    iter: Chars<'a>,
}

unsafe impl<'a> Sync for Drain<'a> {}
unsafe impl<'a> Send for Drain<'a> {}

#[unstable(feature = "drain", reason = "recently added", issue = "27711")]
impl<'a> Drop for Drain<'a> {
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

#[unstable(feature = "drain", reason = "recently added", issue = "27711")]
impl<'a> Iterator for Drain<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[unstable(feature = "drain", reason = "recently added", issue = "27711")]
impl<'a> DoubleEndedIterator for Drain<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        self.iter.next_back()
    }
}
