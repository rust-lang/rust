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

use core::prelude::*;

use core::borrow::{Cow, IntoCow};
use core::default::Default;
use core::fmt;
use core::mem;
use core::ptr;
use core::ops;
// FIXME: ICE's abound if you import the `Slice` type while importing `Slice` trait
use core::raw::Slice as RawSlice;

use hash;
use slice::CloneSliceAllocPrelude;
use str;
use str::{CharRange, CowString, FromStr, StrAllocating, Owned};
use vec::{DerefVec, Vec, as_vec};

/// A growable string stored as a UTF-8 encoded buffer.
#[deriving(Clone, PartialOrd, Eq, Ord)]
#[stable]
pub struct String {
    vec: Vec<u8>,
}

impl String {
    /// Creates a new string buffer initialized with the empty string.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::new();
    /// ```
    #[inline]
    #[stable]
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
    #[stable]
    pub fn with_capacity(capacity: uint) -> String {
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
    #[experimental = "needs investigation to see if to_string() can match perf"]
    pub fn from_str(string: &str) -> String {
        String { vec: string.as_bytes().to_vec() }
    }

    /// Returns the vector as a string buffer, if possible, taking care not to
    /// copy it.
    ///
    /// Returns `Err` with the original vector if the vector contains invalid
    /// UTF-8.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let hello_vec = vec![104, 101, 108, 108, 111];
    /// let s = String::from_utf8(hello_vec);
    /// assert_eq!(s, Ok("hello".to_string()));
    ///
    /// let invalid_vec = vec![240, 144, 128];
    /// let s = String::from_utf8(invalid_vec);
    /// assert_eq!(s, Err(vec![240, 144, 128]));
    /// ```
    #[inline]
    #[unstable = "error type may change"]
    pub fn from_utf8(vec: Vec<u8>) -> Result<String, Vec<u8>> {
        if str::is_utf8(vec.as_slice()) {
            Ok(String { vec: vec })
        } else {
            Err(vec)
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
    /// assert_eq!(output.as_slice(), "Hello \uFFFDWorld");
    /// ```
    #[unstable = "return type may change"]
    pub fn from_utf8_lossy<'a>(v: &'a [u8]) -> CowString<'a> {
        if str::is_utf8(v) {
            return Cow::Borrowed(unsafe { mem::transmute(v) })
        }

        static TAG_CONT_U8: u8 = 128u8;
        static REPLACEMENT: &'static [u8] = b"\xEF\xBF\xBD"; // U+FFFD in UTF-8
        let mut i = 0;
        let total = v.len();
        fn unsafe_get(xs: &[u8], i: uint) -> u8 {
            unsafe { *xs.unsafe_get(i) }
        }
        fn safe_get(xs: &[u8], i: uint, total: uint) -> u8 {
            if i >= total {
                0
            } else {
                unsafe_get(xs, i)
            }
        }

        let mut res = String::with_capacity(total);

        if i > 0 {
            unsafe {
                res.as_mut_vec().push_all(v[..i])
            };
        }

        // subseqidx is the index of the first byte of the subsequence we're looking at.
        // It's used to copy a bunch of contiguous good codepoints at once instead of copying
        // them one by one.
        let mut subseqidx = 0;

        while i < total {
            let i_ = i;
            let byte = unsafe_get(v, i);
            i += 1;

            macro_rules! error(() => ({
                unsafe {
                    if subseqidx != i_ {
                        res.as_mut_vec().push_all(v[subseqidx..i_]);
                    }
                    subseqidx = i;
                    res.as_mut_vec().push_all(REPLACEMENT);
                }
            }))

            if byte < 128u8 {
                // subseqidx handles this
            } else {
                let w = str::utf8_char_width(byte);

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
                res.as_mut_vec().push_all(v[subseqidx..total])
            };
        }
        Cow::Owned(res.into_string())
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
    /// assert_eq!(String::from_utf16(v), Some("𝄞music".to_string()));
    ///
    /// // 𝄞mu<invalid>ic
    /// v[4] = 0xD800;
    /// assert_eq!(String::from_utf16(v), None);
    /// ```
    #[unstable = "error value in return may change"]
    pub fn from_utf16(v: &[u16]) -> Option<String> {
        let mut s = String::with_capacity(v.len());
        for c in str::utf16_items(v) {
            match c {
                str::ScalarValue(c) => s.push(c),
                str::LoneSurrogate(_) => return None
            }
        }
        Some(s)
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
    ///            "𝄞mus\uFFFDic\uFFFD".to_string());
    /// ```
    #[stable]
    pub fn from_utf16_lossy(v: &[u16]) -> String {
        str::utf16_items(v).map(|c| c.to_char_lossy()).collect()
    }

    /// Convert a vector of `char`s to a `String`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let chars = &['h', 'e', 'l', 'l', 'o'];
    /// let s = String::from_chars(chars);
    /// assert_eq!(s.as_slice(), "hello");
    /// ```
    #[inline]
    #[unstable = "may be removed in favor of .collect()"]
    pub fn from_chars(chs: &[char]) -> String {
        chs.iter().map(|c| *c).collect()
    }

    /// Creates a new `String` from a length, capacity, and pointer.
    ///
    /// This is unsafe because:
    /// * We call `Vec::from_raw_parts` to get a `Vec<u8>`;
    /// * We assume that the `Vec` contains valid UTF-8.
    #[inline]
    #[unstable = "function just moved from string::raw"]
    pub unsafe fn from_raw_parts(buf: *mut u8, length: uint, capacity: uint) -> String {
        String {
            vec: Vec::from_raw_parts(buf, length, capacity),
        }
    }

    /// Creates a `String` from a null-terminated `*const u8` buffer.
    ///
    /// This function is unsafe because we dereference memory until we find the
    /// NUL character, which is not guaranteed to be present. Additionally, the
    /// slice is not checked to see whether it contains valid UTF-8
    #[unstable = "just renamed from `mod raw`"]
    pub unsafe fn from_raw_buf(buf: *const u8) -> String {
        String::from_str(str::from_c_str(buf as *const i8))
    }

    /// Creates a `String` from a `*const u8` buffer of the given length.
    ///
    /// This function is unsafe because it blindly assumes the validity of the
    /// pointer `buf` for `len` bytes of memory. This function will copy the
    /// memory from `buf` into a new allocation (owned by the returned
    /// `String`).
    ///
    /// This function is also unsafe because it does not validate that the
    /// buffer is valid UTF-8 encoded data.
    #[unstable = "just renamed from `mod raw`"]
    pub unsafe fn from_raw_buf_len(buf: *const u8, len: uint) -> String {
        String::from_utf8_unchecked(Vec::from_raw_buf(buf, len))
    }

    /// Converts a vector of bytes to a new `String` without checking if
    /// it contains valid UTF-8. This is unsafe because it assumes that
    /// the UTF-8-ness of the vector has already been validated.
    #[inline]
    #[unstable = "awaiting stabilization"]
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
    #[stable]
    pub fn into_bytes(self) -> Vec<u8> {
        self.vec
    }

    /// Creates a string buffer by repeating a character `length` times.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::from_char(5, 'a');
    /// assert_eq!(s.as_slice(), "aaaaa");
    /// ```
    #[inline]
    #[unstable = "may be replaced with iterators, questionable usability, and \
                  the name may change"]
    pub fn from_char(length: uint, ch: char) -> String {
        if length == 0 {
            return String::new()
        }

        let mut buf = String::new();
        buf.push(ch);
        let size = buf.len() * (length - 1);
        buf.reserve_exact(size);
        for _ in range(1, length) {
            buf.push(ch)
        }
        buf
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
    #[unstable = "extra variants of `push`, could possibly be based on iterators"]
    pub fn push_str(&mut self, string: &str) {
        self.vec.push_all(string.as_bytes())
    }

    /// Pushes `ch` onto the given string `count` times.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from_str("foo");
    /// s.grow(5, 'Z');
    /// assert_eq!(s.as_slice(), "fooZZZZZ");
    /// ```
    #[inline]
    #[unstable = "duplicate of iterator-based functionality"]
    pub fn grow(&mut self, count: uint, ch: char) {
        for _ in range(0, count) {
            self.push(ch)
        }
    }

    /// Returns the number of bytes that this string buffer can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = String::with_capacity(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn capacity(&self) -> uint {
        self.vec.capacity()
    }

    /// Deprecated: Renamed to `reserve`.
    #[deprecated = "Renamed to `reserve`"]
    pub fn reserve_additional(&mut self, extra: uint) {
        self.vec.reserve(extra)
    }

    /// Reserves capacity for at least `additional` more bytes to be inserted in the given
    /// `String`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `uint`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::new();
    /// s.reserve(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn reserve(&mut self, additional: uint) {
        self.vec.reserve(additional)
    }

    /// Reserves the minimum capacity for exactly `additional` more bytes to be inserted in the
    /// given `String`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `uint`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::new();
    /// s.reserve(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn reserve_exact(&mut self, additional: uint) {
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
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
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
    #[stable = "function just renamed from push_char"]
    pub fn push(&mut self, ch: char) {
        let cur_len = self.len();
        // This may use up to 4 bytes.
        self.vec.reserve(4);

        unsafe {
            // Attempt to not use an intermediate buffer by just pushing bytes
            // directly onto this string.
            let slice = RawSlice {
                data: self.vec.as_ptr().offset(cur_len as int),
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
    #[stable]
    pub fn as_bytes<'a>(&'a self) -> &'a [u8] {
        self.vec.as_slice()
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
    #[unstable = "the panic conventions for strings are under development"]
    pub fn truncate(&mut self, new_len: uint) {
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
    #[unstable = "this function was just renamed from pop_char"]
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
    /// returns it. Returns `None` if `idx` is out of bounds.
    ///
    /// # Warning
    ///
    /// This is an O(n) operation as it requires copying every element in the
    /// buffer.
    ///
    /// # Panics
    ///
    /// If `idx` does not lie on a character boundary, then this function will
    /// panic.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = String::from_str("foo");
    /// assert_eq!(s.remove(0), Some('f'));
    /// assert_eq!(s.remove(1), Some('o'));
    /// assert_eq!(s.remove(0), Some('o'));
    /// assert_eq!(s.remove(0), None);
    /// ```
    #[unstable = "the panic semantics of this function and return type \
                  may change"]
    pub fn remove(&mut self, idx: uint) -> Option<char> {
        let len = self.len();
        if idx >= len { return None }

        let CharRange { ch, next } = self.char_range_at(idx);
        unsafe {
            ptr::copy_memory(self.vec.as_mut_ptr().offset(idx as int),
                             self.vec.as_ptr().offset(next as int),
                             len - next);
            self.vec.set_len(len - (next - idx));
        }
        Some(ch)
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
    #[unstable = "the panic semantics of this function are uncertain"]
    pub fn insert(&mut self, idx: uint, ch: char) {
        let len = self.len();
        assert!(idx <= len);
        assert!(self.is_char_boundary(idx));
        self.vec.reserve(4);
        let mut bits = [0, ..4];
        let amt = ch.encode_utf8(&mut bits).unwrap();

        unsafe {
            ptr::copy_memory(self.vec.as_mut_ptr().offset((idx + amt) as int),
                             self.vec.as_ptr().offset(idx as int),
                             len - idx);
            ptr::copy_memory(self.vec.as_mut_ptr().offset(idx as int),
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
    #[unstable = "the name of this method may be changed"]
    pub unsafe fn as_mut_vec<'a>(&'a mut self) -> &'a mut Vec<u8> {
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
    #[stable]
    pub fn len(&self) -> uint { self.vec.len() }

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
    #[stable]
    pub fn clear(&mut self) {
        self.vec.clear()
    }
}

#[experimental = "waiting on FromIterator stabilization"]
impl FromIterator<char> for String {
    fn from_iter<I:Iterator<char>>(iterator: I) -> String {
        let mut buf = String::new();
        buf.extend(iterator);
        buf
    }
}

#[experimental = "waiting on FromIterator stabilization"]
impl<'a> FromIterator<&'a str> for String {
    fn from_iter<I:Iterator<&'a str>>(iterator: I) -> String {
        let mut buf = String::new();
        buf.extend(iterator);
        buf
    }
}

#[experimental = "waiting on Extend stabilization"]
impl Extend<char> for String {
    fn extend<I:Iterator<char>>(&mut self, mut iterator: I) {
        let (lower_bound, _) = iterator.size_hint();
        self.reserve(lower_bound);
        for ch in iterator {
            self.push(ch)
        }
    }
}

#[experimental = "waiting on Extend stabilization"]
impl<'a> Extend<&'a str> for String {
    fn extend<I: Iterator<&'a str>>(&mut self, mut iterator: I) {
        // A guess that at least one byte per iterator element will be needed.
        let (lower_bound, _) = iterator.size_hint();
        self.reserve(lower_bound);
        for s in iterator {
            self.push_str(s)
        }
    }
}

impl PartialEq for String {
    #[inline]
    fn eq(&self, other: &String) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &String) -> bool { PartialEq::ne(&**self, &**other) }
}

macro_rules! impl_eq {
    ($lhs:ty, $rhs: ty) => {
        impl<'a> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { PartialEq::ne(&**self, &**other) }
        }

        impl<'a> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool { PartialEq::eq(&**self, &**other) }
            #[inline]
            fn ne(&self, other: &$lhs) -> bool { PartialEq::ne(&**self, &**other) }
        }

    }
}

impl_eq!(String, &'a str)
impl_eq!(CowString<'a>, String)

impl<'a, 'b> PartialEq<&'b str> for CowString<'a> {
    #[inline]
    fn eq(&self, other: &&'b str) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &&'b str) -> bool { PartialEq::ne(&**self, &**other) }
}

impl<'a, 'b> PartialEq<CowString<'a>> for &'b str {
    #[inline]
    fn eq(&self, other: &CowString<'a>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &CowString<'a>) -> bool { PartialEq::ne(&**self, &**other) }
}

#[experimental = "waiting on Str stabilization"]
impl Str for String {
    #[inline]
    #[stable]
    fn as_slice<'a>(&'a self) -> &'a str {
        unsafe {
            mem::transmute(self.vec.as_slice())
        }
    }
}

#[experimental = "waiting on StrAllocating stabilization"]
impl StrAllocating for String {
    #[inline]
    fn into_string(self) -> String {
        self
    }
}

#[stable]
impl Default for String {
    fn default() -> String {
        String::new()
    }
}

#[experimental = "waiting on Show stabilization"]
impl fmt::Show for String {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

#[experimental = "waiting on Hash stabilization"]
impl<H: hash::Writer> hash::Hash<H> for String {
    #[inline]
    fn hash(&self, hasher: &mut H) {
        self.as_slice().hash(hasher)
    }
}

#[allow(deprecated)]
#[deprecated = "Use overloaded `core::cmp::PartialEq`"]
impl<'a, S: Str> Equiv<S> for String {
    #[inline]
    fn equiv(&self, other: &S) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[experimental = "waiting on Add stabilization"]
impl<S: Str> Add<S, String> for String {
    fn add(&self, other: &S) -> String {
        let mut s = String::from_str(self.as_slice());
        s.push_str(other.as_slice());
        return s;
    }
}

impl ops::Slice<uint, str> for String {
    #[inline]
    fn as_slice_<'a>(&'a self) -> &'a str {
        self.as_slice()
    }

    #[inline]
    fn slice_from_or_fail<'a>(&'a self, from: &uint) -> &'a str {
        self[][*from..]
    }

    #[inline]
    fn slice_to_or_fail<'a>(&'a self, to: &uint) -> &'a str {
        self[][..*to]
    }

    #[inline]
    fn slice_or_fail<'a>(&'a self, from: &uint, to: &uint) -> &'a str {
        self[][*from..*to]
    }
}

#[experimental = "waiting on Deref stabilization"]
impl ops::Deref<str> for String {
    fn deref<'a>(&'a self) -> &'a str { self.as_slice() }
}

/// Wrapper type providing a `&String` reference via `Deref`.
#[experimental]
pub struct DerefString<'a> {
    x: DerefVec<'a, u8>
}

impl<'a> Deref<String> for DerefString<'a> {
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
#[experimental]
pub fn as_string<'a>(x: &'a str) -> DerefString<'a> {
    DerefString { x: as_vec(x.as_bytes()) }
}

impl FromStr for String {
    #[inline]
    fn from_str(s: &str) -> Option<String> {
        Some(String::from_str(s))
    }
}

/// Trait for converting a type to a string, consuming it in the process.
pub trait IntoString {
    /// Consume and convert to a string.
    fn into_string(self) -> String;
}

/// A generic trait for converting a value to a string
pub trait ToString {
    /// Converts the value of `self` to an owned string
    fn to_string(&self) -> String;
}

impl<T: fmt::Show> ToString for T {
    fn to_string(&self) -> String {
        let mut buf = Vec::<u8>::new();
        let _ = format_args!(|args| fmt::write(&mut buf, args), "{}", self);
        String::from_utf8(buf).unwrap()
    }
}

impl IntoCow<'static, String, str> for String {
    fn into_cow(self) -> CowString<'static> {
        Cow::Owned(self)
    }
}

impl<'a> IntoCow<'a, String, str> for &'a str {
    fn into_cow(self) -> CowString<'a> {
        Cow::Borrowed(self)
    }
}

/// Unsafe operations
#[deprecated]
pub mod raw {
    use super::String;
    use vec::Vec;

    /// Creates a new `String` from a length, capacity, and pointer.
    ///
    /// This is unsafe because:
    /// * We call `Vec::from_raw_parts` to get a `Vec<u8>`;
    /// * We assume that the `Vec` contains valid UTF-8.
    #[inline]
    #[deprecated = "renamed to String::from_raw_parts"]
    pub unsafe fn from_parts(buf: *mut u8, length: uint, capacity: uint) -> String {
        String::from_raw_parts(buf, length, capacity)
    }

    /// Creates a `String` from a `*const u8` buffer of the given length.
    ///
    /// This function is unsafe because of two reasons:
    ///
    /// * A raw pointer is dereferenced and transmuted to `&[u8]`;
    /// * The slice is not checked to see whether it contains valid UTF-8.
    #[deprecated = "renamed to String::from_raw_buf_len"]
    pub unsafe fn from_buf_len(buf: *const u8, len: uint) -> String {
        String::from_raw_buf_len(buf, len)
    }

    /// Creates a `String` from a null-terminated `*const u8` buffer.
    ///
    /// This function is unsafe because we dereference memory until we find the NUL character,
    /// which is not guaranteed to be present. Additionally, the slice is not checked to see
    /// whether it contains valid UTF-8
    #[deprecated = "renamed to String::from_raw_buf"]
    pub unsafe fn from_buf(buf: *const u8) -> String {
        String::from_raw_buf(buf)
    }

    /// Converts a vector of bytes to a new `String` without checking if
    /// it contains valid UTF-8. This is unsafe because it assumes that
    /// the UTF-8-ness of the vector has already been validated.
    #[inline]
    #[deprecated = "renamed to String::from_utf8_unchecked"]
    pub unsafe fn from_utf8(bytes: Vec<u8>) -> String {
        String::from_utf8_unchecked(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use test::Bencher;

    use slice::CloneSliceAllocPrelude;
    use str::{Str, StrPrelude};
    use str;
    use super::{as_string, String, ToString};
    use vec::Vec;

    #[test]
    fn test_as_string() {
        let x = "foo";
        assert_eq!(x, as_string(x).as_slice());
    }

    #[test]
    fn test_from_str() {
      let owned: Option<::std::string::String> = from_str("string");
      assert_eq!(owned.as_ref().map(|s| s.as_slice()), Some("string"));
    }

    #[test]
    fn test_from_utf8() {
        let xs = b"hello".to_vec();
        assert_eq!(String::from_utf8(xs), Ok(String::from_str("hello")));

        let xs = "ศไทย中华Việt Nam".as_bytes().to_vec();
        assert_eq!(String::from_utf8(xs), Ok(String::from_str("ศไทย中华Việt Nam")));

        let xs = b"hello\xFF".to_vec();
        assert_eq!(String::from_utf8(xs),
                   Err(b"hello\xFF".to_vec()));
    }

    #[test]
    fn test_from_utf8_lossy() {
        let xs = b"hello";
        let ys: str::CowString = "hello".into_cow();
        assert_eq!(String::from_utf8_lossy(xs), ys);

        let xs = "ศไทย中华Việt Nam".as_bytes();
        let ys: str::CowString = "ศไทย中华Việt Nam".into_cow();
        assert_eq!(String::from_utf8_lossy(xs), ys);

        let xs = b"Hello\xC2 There\xFF Goodbye";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("Hello\uFFFD There\uFFFD Goodbye").into_cow());

        let xs = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("Hello\uFFFD\uFFFD There\uFFFD Goodbye").into_cow());

        let xs = b"\xF5foo\xF5\x80bar";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("\uFFFDfoo\uFFFD\uFFFDbar").into_cow());

        let xs = b"\xF1foo\xF1\x80bar\xF1\x80\x80baz";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("\uFFFDfoo\uFFFDbar\uFFFDbaz").into_cow());

        let xs = b"\xF4foo\xF4\x80bar\xF4\xBFbaz";
        assert_eq!(String::from_utf8_lossy(xs),
                   String::from_str("\uFFFDfoo\uFFFDbar\uFFFD\uFFFDbaz").into_cow());

        let xs = b"\xF0\x80\x80\x80foo\xF0\x90\x80\x80bar";
        assert_eq!(String::from_utf8_lossy(xs), String::from_str("\uFFFD\uFFFD\uFFFD\uFFFD\
                                               foo\U00010000bar").into_cow());

        // surrogates
        let xs = b"\xED\xA0\x80foo\xED\xBF\xBFbar";
        assert_eq!(String::from_utf8_lossy(xs), String::from_str("\uFFFD\uFFFD\uFFFDfoo\
                                               \uFFFD\uFFFD\uFFFDbar").into_cow());
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
             (String::from_str("\U00020000"),
              vec![0xD840, 0xDC00])];

        for p in pairs.iter() {
            let (s, u) = (*p).clone();
            let s_as_utf16 = s.utf16_units().collect::<Vec<u16>>();
            let u_as_string = String::from_utf16(u.as_slice()).unwrap();

            assert!(str::is_utf16(u.as_slice()));
            assert_eq!(s_as_utf16, u);

            assert_eq!(u_as_string, s);
            assert_eq!(String::from_utf16_lossy(u.as_slice()), s);

            assert_eq!(String::from_utf16(s_as_utf16.as_slice()).unwrap(), s);
            assert_eq!(u_as_string.utf16_units().collect::<Vec<u16>>(), u);
        }
    }

    #[test]
    fn test_utf16_invalid() {
        // completely positive cases tested above.
        // lead + eof
        assert_eq!(String::from_utf16(&[0xD800]), None);
        // lead + lead
        assert_eq!(String::from_utf16(&[0xD800, 0xD800]), None);

        // isolated trail
        assert_eq!(String::from_utf16(&[0x0061, 0xDC00]), None);

        // general
        assert_eq!(String::from_utf16(&[0xD800, 0xd801, 0xdc8b, 0xD800]), None);
    }

    #[test]
    fn test_from_utf16_lossy() {
        // completely positive cases tested above.
        // lead + eof
        assert_eq!(String::from_utf16_lossy(&[0xD800]), String::from_str("\uFFFD"));
        // lead + lead
        assert_eq!(String::from_utf16_lossy(&[0xD800, 0xD800]), String::from_str("\uFFFD\uFFFD"));

        // isolated trail
        assert_eq!(String::from_utf16_lossy(&[0x0061, 0xDC00]), String::from_str("a\uFFFD"));

        // general
        assert_eq!(String::from_utf16_lossy(&[0xD800, 0xd801, 0xdc8b, 0xD800]),
                   String::from_str("\uFFFD𐒋\uFFFD"));
    }

    #[test]
    fn test_from_buf_len() {
        unsafe {
            let a = vec![65u8, 65, 65, 65, 65, 65, 65, 0];
            assert_eq!(super::raw::from_buf_len(a.as_ptr(), 3), String::from_str("AAA"));
        }
    }

    #[test]
    fn test_from_buf() {
        unsafe {
            let a = vec![65, 65, 65, 65, 65, 65, 65, 0];
            let b = a.as_ptr();
            let c = super::raw::from_buf(b);
            assert_eq!(c, String::from_str("AAAAAAA"));
        }
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
        assert_eq!(s.slice_from(0), "");
        s.push_str("abc");
        assert_eq!(s.slice_from(0), "abc");
        s.push_str("ประเทศไทย中华Việt Nam");
        assert_eq!(s.slice_from(0), "abcประเทศไทย中华Việt Nam");
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
        let mut s = String::from_str("\u00FC"); // ü
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
        let b = b + String::from_str("2");
        assert_eq!(b.len(), 7);
        assert_eq!(b, "1234522");
    }

    #[test]
    fn remove() {
        let mut s = "ศไทย中华Việt Nam; foobar".to_string();;
        assert_eq!(s.remove(0), Some('ศ'));
        assert_eq!(s.len(), 33);
        assert_eq!(s, "ไทย中华Việt Nam; foobar");
        assert_eq!(s.remove(33), None);
        assert_eq!(s.remove(300), None);
        assert_eq!(s.remove(17), Some('ệ'));
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
        assert_eq!("foobar", s[]);
        assert_eq!("foo", s[..3]);
        assert_eq!("bar", s[3..]);
        assert_eq!("oob", s[1..4]);
    }

    #[test]
    fn test_simple_types() {
        assert_eq!(1i.to_string(), "1");
        assert_eq!((-1i).to_string(), "-1");
        assert_eq!(200u.to_string(), "200");
        assert_eq!(2u8.to_string(), "2");
        assert_eq!(true.to_string(), "true");
        assert_eq!(false.to_string(), "false");
        assert_eq!(().to_string(), "()");
        assert_eq!(("hi".to_string()).to_string(), "hi");
    }

    #[test]
    fn test_vectors() {
        let x: Vec<int> = vec![];
        assert_eq!(x.to_string(), "[]");
        assert_eq!((vec![1i]).to_string(), "[1]");
        assert_eq!((vec![1i, 2, 3]).to_string(), "[1, 2, 3]");
        assert!((vec![vec![], vec![1i], vec![1i, 1]]).to_string() ==
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
        let s = Vec::from_elem(100, 0xF5u8);
        b.iter(|| {
            let _ = String::from_utf8_lossy(s.as_slice());
        });
    }
}
