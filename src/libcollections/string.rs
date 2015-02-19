// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

//! An owned, growable string that enforces that its contents are valid UTF-8.

#![stable(feature = "rust1", since = "1.0.0")]

use core::prelude::*;

use core::default::Default;
use core::error::Error;
use core::fmt;
use core::hash;
use core::iter::{IntoIterator, FromIterator};
use core::mem;
use core::ops::{self, Deref, Add, Index};
use core::ptr;
use core::raw::Slice as RawSlice;
use unicode::str as unicode_str;
use unicode::str::Utf16Item;

use borrow::{Cow, IntoCow};
use str::{self, CharRange, FromStr, Utf8Error};
use vec::{DerefVec, Vec, as_vec};

/// A growable string stored as a UTF-8 encoded buffer.
#[derive(Clone, PartialOrd, Eq, Ord)]
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> String {
        String {
            vec: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new string buffer from the given string.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from_str("hello");
    /// assert_eq!(s.as_slice(), "hello");
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "needs investigation to see if to_string() can match perf")]
    pub fn from_str(string: &str) -> String {
        String { vec: ::slice::SliceExt::to_vec(string.as_bytes()) }
    }

    /// Returns the vector as a string buffer, if possible, taking care not to
    /// copy it.
    ///
    /// # Failure
    ///
    /// If the given vector is not valid UTF-8, then the original vector and the
    /// corresponding error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::str::Utf8Error;
    ///
    /// let hello_vec = vec![104, 101, 108, 108, 111];
    /// let s = String::from_utf8(hello_vec).unwrap();
    /// assert_eq!(s, "hello");
    ///
    /// let invalid_vec = vec![240, 144, 128];
    /// let s = String::from_utf8(invalid_vec).err().unwrap();
    /// assert_eq!(s.utf8_error(), Utf8Error::TooShort);
    /// assert_eq!(s.into_bytes(), vec![240, 144, 128]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf8(vec: Vec<u8>) -> Result<String, FromUtf8Error> {
        match str::from_utf8(&vec) {
            Ok(..) => Ok(String { vec: vec }),
            Err(e) => Err(FromUtf8Error { bytes: vec, error: e })
        }
    }

    /// Converts a vector of bytes to a new UTF-8 string.
    /// Any invalid UTF-8 sequences are replaced with U+FFFD REPLACEMENT CHARACTER.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let input = b"Hello \xF0\x90\x80World";
    /// let output = String::from_utf8_lossy(input);
    /// assert_eq!(output.as_slice(), "Hello \u{FFFD}World");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_utf8_lossy<'a>(v: &'a [u8]) -> Cow<'a, str> {
        let mut i = 0;
        match str::from_utf8(v) {
            Ok(s) => return Cow::Borrowed(s),
            Err(e) => {
                if let Utf8Error::InvalidByte(firstbad) = e {
                    i = firstbad;
                }
            }
        }

        static TAG_CONT_U8: u8 = 128u8;
        static REPLACEMENT: &'static [u8] = b"\xEF\xBF\xBD"; // U+FFFD in UTF-8
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

        // subseqidx is the index of the first byte of the subsequence we're looking at.
        // It's used to copy a bunch of contiguous good codepoints at once instead of copying
        // them one by one.
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

            if byte < 128u8 {
                // subseqidx handles this
            } else {
                let w = unicode_str::utf8_char_width(byte);

                match w {
                    2 => {
                        if safe_get(v, i, total) & 192u8 != TAG_CONT_U8 {
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
                        if safe_get(v, i, total) & 192u8 != TAG_CONT_U8 {
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
                        if safe_get(v, i, total) & 192u8 != TAG_CONT_U8 {
                            error!();
                            continue;
                        }
                        i += 1;
                        if safe_get(v, i, total) & 192u8 != TAG_CONT_U8 {
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
    /// ```rust
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
        let mut s = String::with_capacity(v.len());
        for c in unicode_str::utf16_items(v) {
            match c {
                Utf16Item::ScalarValue(c) => s.push(c),
                Utf16Item::LoneSurrogate(_) => return Err(FromUtf16Error(())),
            }
        }
        Ok(s)
    }

    /// Decode a UTF-16 encoded vector `v` into a string, replacing
    /// invalid data with the replacement character (U+FFFD).
    ///
    /// # Examples
    ///
    /// ```rust
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
        unicode_str::utf16_items(v).map(|c| c.to_char_lossy()).collect()
    }

    /// Creates a new `String` from a length, capacity, and pointer.
    ///
    /// This is unsafe because:
    /// * We call `Vec::from_raw_parts` to get a `Vec<u8>`;
    /// * We assume that the `Vec` contains valid UTF-8.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_raw_parts(buf: *mut u8, length: usize, capacity: usize) -> String {
        String {
            vec: Vec::from_raw_parts(buf, length, capacity),
        }
    }

    /// Converts a vector of bytes to a new `String` without checking if
    /// it contains valid UTF-8. This is unsafe because it assumes that
    /// the UTF-8-ness of the vector has already been validated.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_utf8_unchecked(bytes: Vec<u8>) -> String {
        String { vec: bytes }
    }

    /// Return the underlying byte buffer, encoded as UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from_str("hello");
    /// let bytes = s.into_bytes();
    /// assert_eq!(bytes, vec![104, 101, 108, 108, 111]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_bytes(self) -> Vec<u8> {
        self.vec
    }

    /// Pushes the given string onto this string buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from_str("foo");
    /// s.push_str("bar");
    /// assert_eq!(s.as_slice(), "foobar");
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
    /// s.reserve(10);
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
    /// let mut s = String::from_str("foo");
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
    /// let mut s = String::from_str("abc");
    /// s.push('1');
    /// s.push('2');
    /// s.push('3');
    /// assert_eq!(s.as_slice(), "abc123");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, ch: char) {
        if (ch as u32) < 0x80 {
            self.vec.push(ch as u8);
            return;
        }

        let cur_len = self.len();
        // This may use up to 4 bytes.
        self.vec.reserve(4);

        unsafe {
            // Attempt to not use an intermediate buffer by just pushing bytes
            // directly onto this string.
            let slice = RawSlice {
                data: self.vec.as_ptr().offset(cur_len as isize),
                len: 4,
            };
            let used = ch.encode_utf8(mem::transmute(slice)).unwrap_or(0);
            self.vec.set_len(cur_len + used);
        }
    }

    /// Works with the underlying buffer as a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from_str("hello");
    /// let b: &[_] = &[104, 101, 108, 108, 111];
    /// assert_eq!(s.as_bytes(), b);
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
    /// let mut s = String::from_str("hello");
    /// s.truncate(2);
    /// assert_eq!(s.as_slice(), "he");
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
    /// let mut s = String::from_str("foo");
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

        let CharRange {ch, next} = self.char_range_at_reverse(len);
        unsafe {
            self.vec.set_len(next);
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
    /// let mut s = String::from_str("foo");
    /// assert_eq!(s.remove(0), 'f');
    /// assert_eq!(s.remove(1), 'o');
    /// assert_eq!(s.remove(0), 'o');
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, idx: usize) -> char {
        let len = self.len();
        assert!(idx <= len);

        let CharRange { ch, next } = self.char_range_at(idx);
        unsafe {
            ptr::copy_memory(self.vec.as_mut_ptr().offset(idx as isize),
                             self.vec.as_ptr().offset(next as isize),
                             len - next);
            self.vec.set_len(len - (next - idx));
        }
        ch
    }

    /// Insert a character into the string buffer at byte position `idx`.
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
            ptr::copy_memory(self.vec.as_mut_ptr().offset((idx + amt) as isize),
                             self.vec.as_ptr().offset(idx as isize),
                             len - idx);
            ptr::copy_memory(self.vec.as_mut_ptr().offset(idx as isize),
                             bits.as_ptr(),
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
    /// let mut s = String::from_str("hello");
    /// unsafe {
    ///     let vec = s.as_mut_vec();
    ///     assert!(vec == &mut vec![104, 101, 108, 108, 111]);
    ///     vec.reverse();
    /// }
    /// assert_eq!(s.as_slice(), "olleh");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        &mut self.vec
    }

    /// Return the number of bytes in this string.
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
}

impl FromUtf8Error {
    /// Consume this error, returning the bytes that were attempted to make a
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
impl Error for FromUtf8Error {
    fn description(&self) -> &str { "invalid utf-8" }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for FromUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt("invalid utf-16: lone surrogate found", f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for FromUtf16Error {
    fn description(&self) -> &str { "invalid utf-16" }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromIterator<char> for String {
    fn from_iter<I: IntoIterator<Item=char>>(iter: I) -> String {
        let mut buf = String::new();
        buf.extend(iter);
        buf
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> FromIterator<&'a str> for String {
    fn from_iter<I: IntoIterator<Item=&'a str>>(iter: I) -> String {
        let mut buf = String::new();
        buf.extend(iter);
        buf
    }
}

#[unstable(feature = "collections",
           reason = "waiting on Extend stabilization")]
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

#[unstable(feature = "collections",
           reason = "waiting on Extend stabilization")]
impl<'a> Extend<&'a str> for String {
    fn extend<I: IntoIterator<Item=&'a str>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        // A guess that at least one byte per iterator element will be needed.
        let (lower_bound, _) = iterator.size_hint();
        self.reserve(lower_bound);
        for s in iterator {
            self.push_str(s)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for String {
    #[inline]
    fn eq(&self, other: &String) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &String) -> bool { PartialEq::ne(&**self, &**other) }
}

macro_rules! impl_eq {
    ($lhs:ty, $rhs: ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { PartialEq::ne(&**self, &**other) }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$lhs) -> bool { PartialEq::ne(&**self, &**other) }
        }

    }
}

impl_eq! { String, &'a str }
impl_eq! { Cow<'a, str>, String }

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b> PartialEq<&'b str> for Cow<'a, str> {
    #[inline]
    fn eq(&self, other: &&'b str) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &&'b str) -> bool { PartialEq::ne(&**self, &**other) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b> PartialEq<Cow<'a, str>> for &'b str {
    #[inline]
    fn eq(&self, other: &Cow<'a, str>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Cow<'a, str>) -> bool { PartialEq::ne(&**self, &**other) }
}

#[unstable(feature = "collections", reason = "waiting on Str stabilization")]
impl Str for String {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_slice(&self) -> &str {
        unsafe { mem::transmute(&*self.vec) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for String {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
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

#[unstable(feature = "collections", reason = "waiting on Hash stabilization")]
#[cfg(stage0)]
impl<H: hash::Writer + hash::Hasher> hash::Hash<H> for String {
    #[inline]
    fn hash(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(not(stage0))]
impl hash::Hash for String {
    #[inline]
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}

#[unstable(feature = "collections",
           reason = "recent addition, needs more experience")]
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
    fn index(&self, index: &ops::Range<usize>) -> &str {
        &self[..][*index]
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeTo<usize>> for String {
    type Output = str;
    #[inline]
    fn index(&self, index: &ops::RangeTo<usize>) -> &str {
        &self[..][*index]
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeFrom<usize>> for String {
    type Output = str;
    #[inline]
    fn index(&self, index: &ops::RangeFrom<usize>) -> &str {
        &self[..][*index]
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeFull> for String {
    type Output = str;
    #[inline]
    fn index(&self, _index: &ops::RangeFull) -> &str {
        unsafe { mem::transmute(&*self.vec) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for String {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        unsafe { mem::transmute(&self.vec[..]) }
    }
}

/// Wrapper type providing a `&String` reference via `Deref`.
#[unstable(feature = "collections")]
pub struct DerefString<'a> {
    x: DerefVec<'a, u8>
}

impl<'a> Deref for DerefString<'a> {
    type Target = String;

    #[inline]
    fn deref<'b>(&'b self) -> &'b String {
        unsafe { mem::transmute(&*self.x) }
    }
}

/// Convert a string slice to a wrapper type providing a `&String` reference.
///
/// # Examples
///
/// ```
/// use std::string::as_string;
///
/// fn string_consumer(s: String) {
///     assert_eq!(s, "foo".to_string());
/// }
///
/// let string = as_string("foo").clone();
/// string_consumer(string);
/// ```
#[unstable(feature = "collections")]
pub fn as_string<'a>(x: &'a str) -> DerefString<'a> {
    DerefString { x: as_vec(x.as_bytes()) }
}

#[unstable(feature = "collections", reason = "associated error type may change")]
impl FromStr for String {
    type Err = ();
    #[inline]
    fn from_str(s: &str) -> Result<String, ()> {
        Ok(String::from_str(s))
    }
}

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
impl IntoCow<'static, str> for String {
    #[inline]
    fn into_cow(self) -> Cow<'static, str> {
        Cow::Owned(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> IntoCow<'a, str> for &'a str {
    #[inline]
    fn into_cow(self) -> Cow<'a, str> {
        Cow::Borrowed(self)
    }
}

impl<'a> Str for Cow<'a, str> {
    #[inline]
    fn as_slice<'b>(&'b self) -> &'b str {
        &**self
    }
}

/// A clone-on-write string
#[deprecated(since = "1.0.0", reason = "use Cow<'a, str> instead")]
#[stable(feature = "rust1", since = "1.0.0")]
pub type CowString<'a> = Cow<'a, str>;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Write for String {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use test::Bencher;

    use str::Utf8Error;
    use core::iter::repeat;
    use super::{as_string, CowString};

    #[test]
    fn test_as_string() {
        let x = "foo";
        assert_eq!(x, &**as_string(x));
    }

    #[test]
    fn test_from_str() {
      let owned: Option<::std::string::String> = "string".parse().ok();
      assert_eq!(owned.as_ref().map(|s| &**s), Some("string"));
    }

    #[test]
    fn test_unsized_to_string() {
        let s: &str = "abc";
        let _: String = (*s).to_string();
    }

    #[test]
    fn test_from_utf8() {
        let xs = b"hello".to_vec();
        assert_eq!(String::from_utf8(xs).unwrap(),
                   String::from_str("hello"));

        let xs = "ศไทย中华Việt Nam".as_bytes().to_vec();
        assert_eq!(String::from_utf8(xs).unwrap(),
                   String::from_str("ศไทย中华Việt Nam"));

        let xs = b"hello\xFF".to_vec();
        let err = String::from_utf8(xs).err().unwrap();
        assert_eq!(err.utf8_error(), Utf8Error::TooShort);
        assert_eq!(err.into_bytes(), b"hello\xff".to_vec());
    }

    #[test]
    fn test_from_utf8_lossy() {
        let xs = b"hello";
        let ys: CowString = "hello".into_cow();
        assert_eq!(String::from_utf8_lossy(xs), ys);

        let xs = "ศไทย中华Việt Nam".as_bytes();
        let ys: CowString = "ศไทย中华Việt Nam".into_cow();
        assert_eq!(String::from_utf8_lossy(xs), ys);

        let xs = b"Hello\xC2 There\xFF Goodbye";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("Hello\u{FFFD} There\u{FFFD} Goodbye").into_cow());

        let xs = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("Hello\u{FFFD}\u{FFFD} There\u{FFFD} Goodbye").into_cow());

        let xs = b"\xF5foo\xF5\x80bar";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("\u{FFFD}foo\u{FFFD}\u{FFFD}bar").into_cow());

        let xs = b"\xF1foo\xF1\x80bar\xF1\x80\x80baz";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("\u{FFFD}foo\u{FFFD}bar\u{FFFD}baz").into_cow());

        let xs = b"\xF4foo\xF4\x80bar\xF4\xBFbaz";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("\u{FFFD}foo\u{FFFD}bar\u{FFFD}\u{FFFD}baz").into_cow());

        let xs = b"\xF0\x80\x80\x80foo\xF0\x90\x80\x80bar";
        assert_eq!(String::from_utf8_lossy(xs), String::from_str("\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\
                                               foo\u{10000}bar").into_cow());

        // surrogates
        let xs = b"\xED\xA0\x80foo\xED\xBF\xBFbar";
        assert_eq!(String::from_utf8_lossy(xs), String::from_str("\u{FFFD}\u{FFFD}\u{FFFD}foo\
                                               \u{FFFD}\u{FFFD}\u{FFFD}bar").into_cow());
    }

    #[test]
    fn test_from_utf16() {
        let pairs =
            [(String::from_str("𐍅𐌿𐌻𐍆𐌹𐌻𐌰\n"),
              vec![0xd800_u16, 0xdf45_u16, 0xd800_u16, 0xdf3f_u16,
                0xd800_u16, 0xdf3b_u16, 0xd800_u16, 0xdf46_u16,
                0xd800_u16, 0xdf39_u16, 0xd800_u16, 0xdf3b_u16,
                0xd800_u16, 0xdf30_u16, 0x000a_u16]),

             (String::from_str("𐐒𐑉𐐮𐑀𐐲𐑋 𐐏𐐲𐑍\n"),
              vec![0xd801_u16, 0xdc12_u16, 0xd801_u16,
                0xdc49_u16, 0xd801_u16, 0xdc2e_u16, 0xd801_u16,
                0xdc40_u16, 0xd801_u16, 0xdc32_u16, 0xd801_u16,
                0xdc4b_u16, 0x0020_u16, 0xd801_u16, 0xdc0f_u16,
                0xd801_u16, 0xdc32_u16, 0xd801_u16, 0xdc4d_u16,
                0x000a_u16]),

             (String::from_str("𐌀𐌖𐌋𐌄𐌑𐌉·𐌌𐌄𐌕𐌄𐌋𐌉𐌑\n"),
              vec![0xd800_u16, 0xdf00_u16, 0xd800_u16, 0xdf16_u16,
                0xd800_u16, 0xdf0b_u16, 0xd800_u16, 0xdf04_u16,
                0xd800_u16, 0xdf11_u16, 0xd800_u16, 0xdf09_u16,
                0x00b7_u16, 0xd800_u16, 0xdf0c_u16, 0xd800_u16,
                0xdf04_u16, 0xd800_u16, 0xdf15_u16, 0xd800_u16,
                0xdf04_u16, 0xd800_u16, 0xdf0b_u16, 0xd800_u16,
                0xdf09_u16, 0xd800_u16, 0xdf11_u16, 0x000a_u16 ]),

             (String::from_str("𐒋𐒘𐒈𐒑𐒛𐒒 𐒕𐒓 𐒈𐒚𐒍 𐒏𐒜𐒒𐒖𐒆 𐒕𐒆\n"),
              vec![0xd801_u16, 0xdc8b_u16, 0xd801_u16, 0xdc98_u16,
                0xd801_u16, 0xdc88_u16, 0xd801_u16, 0xdc91_u16,
                0xd801_u16, 0xdc9b_u16, 0xd801_u16, 0xdc92_u16,
                0x0020_u16, 0xd801_u16, 0xdc95_u16, 0xd801_u16,
                0xdc93_u16, 0x0020_u16, 0xd801_u16, 0xdc88_u16,
                0xd801_u16, 0xdc9a_u16, 0xd801_u16, 0xdc8d_u16,
                0x0020_u16, 0xd801_u16, 0xdc8f_u16, 0xd801_u16,
                0xdc9c_u16, 0xd801_u16, 0xdc92_u16, 0xd801_u16,
                0xdc96_u16, 0xd801_u16, 0xdc86_u16, 0x0020_u16,
                0xd801_u16, 0xdc95_u16, 0xd801_u16, 0xdc86_u16,
                0x000a_u16 ]),
             // Issue #12318, even-numbered non-BMP planes
             (String::from_str("\u{20000}"),
              vec![0xD840, 0xDC00])];

        for p in &pairs {
            let (s, u) = (*p).clone();
            let s_as_utf16 = s.utf16_units().collect::<Vec<u16>>();
            let u_as_string = String::from_utf16(&u).unwrap();

            assert!(::unicode::str::is_utf16(&u));
            assert_eq!(s_as_utf16, u);

            assert_eq!(u_as_string, s);
            assert_eq!(String::from_utf16_lossy(&u), s);

            assert_eq!(String::from_utf16(&s_as_utf16).unwrap(), s);
            assert_eq!(u_as_string.utf16_units().collect::<Vec<u16>>(), u);
        }
    }

    #[test]
    fn test_utf16_invalid() {
        // completely positive cases tested above.
        // lead + eof
        assert!(String::from_utf16(&[0xD800]).is_err());
        // lead + lead
        assert!(String::from_utf16(&[0xD800, 0xD800]).is_err());

        // isolated trail
        assert!(String::from_utf16(&[0x0061, 0xDC00]).is_err());

        // general
        assert!(String::from_utf16(&[0xD800, 0xd801, 0xdc8b, 0xD800]).is_err());
    }

    #[test]
    fn test_from_utf16_lossy() {
        // completely positive cases tested above.
        // lead + eof
        assert_eq!(String::from_utf16_lossy(&[0xD800]), String::from_str("\u{FFFD}"));
        // lead + lead
        assert_eq!(String::from_utf16_lossy(&[0xD800, 0xD800]),
                   String::from_str("\u{FFFD}\u{FFFD}"));

        // isolated trail
        assert_eq!(String::from_utf16_lossy(&[0x0061, 0xDC00]), String::from_str("a\u{FFFD}"));

        // general
        assert_eq!(String::from_utf16_lossy(&[0xD800, 0xd801, 0xdc8b, 0xD800]),
                   String::from_str("\u{FFFD}𐒋\u{FFFD}"));
    }

    #[test]
    fn test_push_bytes() {
        let mut s = String::from_str("ABC");
        unsafe {
            let mv = s.as_mut_vec();
            mv.push_all(&[b'D']);
        }
        assert_eq!(s, "ABCD");
    }

    #[test]
    fn test_push_str() {
        let mut s = String::new();
        s.push_str("");
        assert_eq!(&s[0..], "");
        s.push_str("abc");
        assert_eq!(&s[0..], "abc");
        s.push_str("ประเทศไทย中华Việt Nam");
        assert_eq!(&s[0..], "abcประเทศไทย中华Việt Nam");
    }

    #[test]
    fn test_push() {
        let mut data = String::from_str("ประเทศไทย中");
        data.push('华');
        data.push('b'); // 1 byte
        data.push('¢'); // 2 byte
        data.push('€'); // 3 byte
        data.push('𤭢'); // 4 byte
        assert_eq!(data, "ประเทศไทย中华b¢€𤭢");
    }

    #[test]
    fn test_pop() {
        let mut data = String::from_str("ประเทศไทย中华b¢€𤭢");
        assert_eq!(data.pop().unwrap(), '𤭢'); // 4 bytes
        assert_eq!(data.pop().unwrap(), '€'); // 3 bytes
        assert_eq!(data.pop().unwrap(), '¢'); // 2 bytes
        assert_eq!(data.pop().unwrap(), 'b'); // 1 bytes
        assert_eq!(data.pop().unwrap(), '华');
        assert_eq!(data, "ประเทศไทย中");
    }

    #[test]
    fn test_str_truncate() {
        let mut s = String::from_str("12345");
        s.truncate(5);
        assert_eq!(s, "12345");
        s.truncate(3);
        assert_eq!(s, "123");
        s.truncate(0);
        assert_eq!(s, "");

        let mut s = String::from_str("12345");
        let p = s.as_ptr();
        s.truncate(3);
        s.push_str("6");
        let p_ = s.as_ptr();
        assert_eq!(p_, p);
    }

    #[test]
    #[should_fail]
    fn test_str_truncate_invalid_len() {
        let mut s = String::from_str("12345");
        s.truncate(6);
    }

    #[test]
    #[should_fail]
    fn test_str_truncate_split_codepoint() {
        let mut s = String::from_str("\u{FC}"); // ü
        s.truncate(1);
    }

    #[test]
    fn test_str_clear() {
        let mut s = String::from_str("12345");
        s.clear();
        assert_eq!(s.len(), 0);
        assert_eq!(s, "");
    }

    #[test]
    fn test_str_add() {
        let a = String::from_str("12345");
        let b = a + "2";
        let b = b + "2";
        assert_eq!(b.len(), 7);
        assert_eq!(b, "1234522");
    }

    #[test]
    fn remove() {
        let mut s = "ศไทย中华Việt Nam; foobar".to_string();;
        assert_eq!(s.remove(0), 'ศ');
        assert_eq!(s.len(), 33);
        assert_eq!(s, "ไทย中华Việt Nam; foobar");
        assert_eq!(s.remove(17), 'ệ');
        assert_eq!(s, "ไทย中华Vit Nam; foobar");
    }

    #[test] #[should_fail]
    fn remove_bad() {
        "ศ".to_string().remove(1);
    }

    #[test]
    fn insert() {
        let mut s = "foobar".to_string();
        s.insert(0, 'ệ');
        assert_eq!(s, "ệfoobar");
        s.insert(6, 'ย');
        assert_eq!(s, "ệfooยbar");
    }

    #[test] #[should_fail] fn insert_bad1() { "".to_string().insert(1, 't'); }
    #[test] #[should_fail] fn insert_bad2() { "ệ".to_string().insert(1, 't'); }

    #[test]
    fn test_slicing() {
        let s = "foobar".to_string();
        assert_eq!("foobar", &s[..]);
        assert_eq!("foo", &s[..3]);
        assert_eq!("bar", &s[3..]);
        assert_eq!("oob", &s[1..4]);
    }

    #[test]
    fn test_simple_types() {
        assert_eq!(1.to_string(), "1");
        assert_eq!((-1).to_string(), "-1");
        assert_eq!(200.to_string(), "200");
        assert_eq!(2u8.to_string(), "2");
        assert_eq!(true.to_string(), "true");
        assert_eq!(false.to_string(), "false");
        assert_eq!(("hi".to_string()).to_string(), "hi");
    }

    #[test]
    fn test_vectors() {
        let x: Vec<i32> = vec![];
        assert_eq!(format!("{:?}", x), "[]");
        assert_eq!(format!("{:?}", vec![1]), "[1]");
        assert_eq!(format!("{:?}", vec![1, 2, 3]), "[1, 2, 3]");
        assert!(format!("{:?}", vec![vec![], vec![1], vec![1, 1]]) ==
               "[[], [1], [1, 1]]");
    }

    #[test]
    fn test_from_iterator() {
        let s = "ศไทย中华Việt Nam".to_string();
        let t = "ศไทย中华";
        let u = "Việt Nam";

        let a: String = s.chars().collect();
        assert_eq!(s, a);

        let mut b = t.to_string();
        b.extend(u.chars());
        assert_eq!(s, b);

        let c: String = vec![t, u].into_iter().collect();
        assert_eq!(s, c);

        let mut d = t.to_string();
        d.extend(vec![u].into_iter());
        assert_eq!(s, d);
    }

    #[bench]
    fn bench_with_capacity(b: &mut Bencher) {
        b.iter(|| {
            String::with_capacity(100)
        });
    }

    #[bench]
    fn bench_push_str(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        b.iter(|| {
            let mut r = String::new();
            r.push_str(s);
        });
    }

    const REPETITIONS: u64 = 10_000;

    #[bench]
    fn bench_push_str_one_byte(b: &mut Bencher) {
        b.bytes = REPETITIONS;
        b.iter(|| {
            let mut r = String::new();
            for _ in 0..REPETITIONS {
                r.push_str("a")
            }
        });
    }

    #[bench]
    fn bench_push_char_one_byte(b: &mut Bencher) {
        b.bytes = REPETITIONS;
        b.iter(|| {
            let mut r = String::new();
            for _ in 0..REPETITIONS {
                r.push('a')
            }
        });
    }

    #[bench]
    fn bench_push_char_two_bytes(b: &mut Bencher) {
        b.bytes = REPETITIONS * 2;
        b.iter(|| {
            let mut r = String::new();
            for _ in 0..REPETITIONS {
                r.push('â')
            }
        });
    }

    #[bench]
    fn from_utf8_lossy_100_ascii(b: &mut Bencher) {
        let s = b"Hello there, the quick brown fox jumped over the lazy dog! \
                  Lorem ipsum dolor sit amet, consectetur. ";

        assert_eq!(100, s.len());
        b.iter(|| {
            let _ = String::from_utf8_lossy(s);
        });
    }

    #[bench]
    fn from_utf8_lossy_100_multibyte(b: &mut Bencher) {
        let s = "𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰".as_bytes();
        assert_eq!(100, s.len());
        b.iter(|| {
            let _ = String::from_utf8_lossy(s);
        });
    }

    #[bench]
    fn from_utf8_lossy_invalid(b: &mut Bencher) {
        let s = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
        b.iter(|| {
            let _ = String::from_utf8_lossy(s);
        });
    }

    #[bench]
    fn from_utf8_lossy_100_invalid(b: &mut Bencher) {
        let s = repeat(0xf5u8).take(100).collect::<Vec<_>>();
        b.iter(|| {
            let _ = String::from_utf8_lossy(&s);
        });
    }

    #[bench]
    fn bench_exact_size_shrink_to_fit(b: &mut Bencher) {
        let s = "Hello there, the quick brown fox jumped over the lazy dog! \
                 Lorem ipsum dolor sit amet, consectetur. ";
        // ensure our operation produces an exact-size string before we benchmark it
        let mut r = String::with_capacity(s.len());
        r.push_str(s);
        assert_eq!(r.len(), r.capacity());
        b.iter(|| {
            let mut r = String::with_capacity(s.len());
            r.push_str(s);
            r.shrink_to_fit();
            r
        });
    }
}
