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

use c_vec::CVec;
use char::Char;
use container::Container;
use fmt;
use io::Writer;
use iter::{Extendable, FromIterator, Iterator, range};
use mem;
use option::{None, Option, Some};
use ptr::RawPtr;
use slice::{OwnedVector, Vector, CloneableVector};
use str::{OwnedStr, Str, StrSlice, StrAllocating};
use str;
use vec::Vec;

/// A growable string stored as a UTF-8 encoded buffer.
#[deriving(Clone, Eq, Ord, TotalEq, TotalOrd)]
pub struct StrBuf {
    vec: Vec<u8>,
}

impl StrBuf {
    /// Creates a new string buffer initialized with the empty string.
    #[inline]
    pub fn new() -> StrBuf {
        StrBuf {
            vec: Vec::new(),
        }
    }

    /// Creates a new string buffer with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: uint) -> StrBuf {
        StrBuf {
            vec: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new string buffer from length, capacity, and a pointer.
    #[inline]
    pub unsafe fn from_raw_parts(length: uint, capacity: uint, ptr: *mut u8) -> StrBuf {
        StrBuf {
            vec: Vec::from_raw_parts(length, capacity, ptr),
        }
    }

    /// Creates a new string buffer from the given string.
    #[inline]
    pub fn from_str(string: &str) -> StrBuf {
        StrBuf {
            vec: Vec::from_slice(string.as_bytes())
        }
    }

    /// Creates a new string buffer from the given owned string, taking care not to copy it.
    #[inline]
    pub fn from_owned_str(string: ~str) -> StrBuf {
        StrBuf {
            vec: string.into_bytes().move_iter().collect(),
        }
    }

    /// Tries to create a new string buffer from the given byte
    /// vector, validating that the vector is UTF-8 encoded.
    #[inline]
    pub fn from_utf8(vec: Vec<u8>) -> Option<StrBuf> {
        if str::is_utf8(vec.as_slice()) {
            Some(StrBuf { vec: vec })
        } else {
            None
        }
    }

    /// Return the underlying byte buffer, encoded as UTF-8.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.vec
    }

    /// Pushes the given string onto this buffer; then, returns `self` so that it can be used
    /// again.
    #[inline]
    pub fn append(mut self, second: &str) -> StrBuf {
        self.push_str(second);
        self
    }

    /// Creates a string buffer by repeating a character `length` times.
    #[inline]
    pub fn from_char(length: uint, ch: char) -> StrBuf {
        if length == 0 {
            return StrBuf::new()
        }

        let mut buf = StrBuf::new();
        buf.push_char(ch);
        let size = buf.len() * length;
        buf.reserve(size);
        for _ in range(1, length) {
            buf.push_char(ch)
        }
        buf
    }

    /// Pushes the given string onto this string buffer.
    #[inline]
    pub fn push_str(&mut self, string: &str) {
        self.vec.push_all(string.as_bytes())
    }

    /// Push `ch` onto the given string `count` times.
    #[inline]
    pub fn grow(&mut self, count: uint, ch: char) {
        for _ in range(0, count) {
            self.push_char(ch)
        }
    }

    /// Returns the number of bytes that this string buffer can hold without reallocating.
    #[inline]
    pub fn byte_capacity(&self) -> uint {
        self.vec.capacity()
    }

    /// Reserves capacity for at least `extra` additional bytes in this string buffer.
    #[inline]
    pub fn reserve_additional(&mut self, extra: uint) {
        self.vec.reserve_additional(extra)
    }

    /// Reserves capacity for at least `capacity` bytes in this string buffer.
    #[inline]
    pub fn reserve(&mut self, capacity: uint) {
        self.vec.reserve(capacity)
    }

    /// Reserves capacity for exactly `capacity` bytes in this string buffer.
    #[inline]
    pub fn reserve_exact(&mut self, capacity: uint) {
        self.vec.reserve_exact(capacity)
    }

    /// Shrinks the capacity of this string buffer to match its length.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit()
    }

    /// Adds the given character to the end of the string.
    #[inline]
    pub fn push_char(&mut self, ch: char) {
        let cur_len = self.len();
        unsafe {
            // This may use up to 4 bytes.
            self.vec.reserve_additional(4);

            // Attempt to not use an intermediate buffer by just pushing bytes
            // directly onto this string.
            let mut c_vector = CVec::new(self.vec.as_mut_ptr().offset(cur_len as int), 4);
            let used = ch.encode_utf8(c_vector.as_mut_slice());
            self.vec.set_len(cur_len + used);
        }
    }

    /// Pushes the given bytes onto this string buffer. This is unsafe because it does not check
    /// to ensure that the resulting string will be valid UTF-8.
    #[inline]
    pub unsafe fn push_bytes(&mut self, bytes: &[u8]) {
        self.vec.push_all(bytes)
    }

    /// Works with the underlying buffer as a byte slice.
    #[inline]
    pub fn as_bytes<'a>(&'a self) -> &'a [u8] {
        self.vec.as_slice()
    }

    /// Shorten a string to the specified length (which must be <= the current length)
    #[inline]
    pub fn truncate(&mut self, len: uint) {
        assert!(self.as_slice().is_char_boundary(len));
        self.vec.truncate(len)
    }

    /// Appends a byte to this string buffer. The caller must preserve the valid UTF-8 property.
    #[inline]
    pub unsafe fn push_byte(&mut self, byte: u8) {
        self.push_bytes([byte])
    }

    /// Removes the last byte from the string buffer and returns it. Returns `None` if this string
    /// buffer is empty.
    ///
    /// The caller must preserve the valid UTF-8 property.
    #[inline]
    pub unsafe fn pop_byte(&mut self) -> Option<u8> {
        let len = self.len();
        if len == 0 {
            return None
        }

        let byte = self.as_slice()[len - 1];
        self.vec.set_len(len - 1);
        Some(byte)
    }

    /// Removes the first byte from the string buffer and returns it. Returns `None` if this string
    /// buffer is empty.
    ///
    /// The caller must preserve the valid UTF-8 property.
    pub unsafe fn shift_byte(&mut self) -> Option<u8> {
        let len = self.len();
        if len == 0 {
            return None
        }

        let byte = self.as_slice()[0];
        *self = self.as_slice().slice(1, len).into_strbuf();
        Some(byte)
    }

    /// Views the string buffer as a mutable sequence of bytes.
    ///
    /// Callers must preserve the valid UTF-8 property.
    pub unsafe fn as_mut_vec<'a>(&'a mut self) -> &'a mut Vec<u8> {
        &mut self.vec
    }
}

impl Container for StrBuf {
    #[inline]
    fn len(&self) -> uint {
        self.vec.len()
    }
}

impl FromIterator<char> for StrBuf {
    fn from_iter<I:Iterator<char>>(iterator: I) -> StrBuf {
        let mut buf = StrBuf::new();
        buf.extend(iterator);
        buf
    }
}

impl Extendable<char> for StrBuf {
    fn extend<I:Iterator<char>>(&mut self, mut iterator: I) {
        for ch in iterator {
            self.push_char(ch)
        }
    }
}

impl Str for StrBuf {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str {
        unsafe {
            mem::transmute(self.vec.as_slice())
        }
    }
}

impl StrAllocating for StrBuf {
    #[inline]
    fn into_owned(self) -> ~str {
        unsafe {
            mem::transmute(self.vec.as_slice().to_owned())
        }
    }

    #[inline]
    fn into_strbuf(self) -> StrBuf { self }
}

impl fmt::Show for StrBuf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<H:Writer> ::hash::Hash<H> for StrBuf {
    #[inline]
    fn hash(&self, hasher: &mut H) {
        self.as_slice().hash(hasher)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use str::{Str, StrSlice};
    use super::StrBuf;

    #[bench]
    fn bench_with_capacity(b: &mut Bencher) {
        b.iter(|| {
            StrBuf::with_capacity(100)
        });
    }

    #[bench]
    fn bench_push_str(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        b.iter(|| {
            let mut r = StrBuf::new();
            r.push_str(s);
        });
    }

    #[test]
    fn test_push_bytes() {
        let mut s = StrBuf::from_str("ABC");
        unsafe {
            s.push_bytes([ 'D' as u8 ]);
        }
        assert_eq!(s.as_slice(), "ABCD");
    }

    #[test]
    fn test_push_str() {
        let mut s = StrBuf::new();
        s.push_str("");
        assert_eq!(s.as_slice().slice_from(0), "");
        s.push_str("abc");
        assert_eq!(s.as_slice().slice_from(0), "abc");
        s.push_str("ประเทศไทย中华Việt Nam");
        assert_eq!(s.as_slice().slice_from(0), "abcประเทศไทย中华Việt Nam");
    }

    #[test]
    fn test_push_char() {
        let mut data = StrBuf::from_str("ประเทศไทย中");
        data.push_char('华');
        data.push_char('b'); // 1 byte
        data.push_char('¢'); // 2 byte
        data.push_char('€'); // 3 byte
        data.push_char('𤭢'); // 4 byte
        assert_eq!(data.as_slice(), "ประเทศไทย中华b¢€𤭢");
    }

    #[test]
    fn test_str_truncate() {
        let mut s = StrBuf::from_str("12345");
        s.truncate(5);
        assert_eq!(s.as_slice(), "12345");
        s.truncate(3);
        assert_eq!(s.as_slice(), "123");
        s.truncate(0);
        assert_eq!(s.as_slice(), "");

        let mut s = StrBuf::from_str("12345");
        let p = s.as_slice().as_ptr();
        s.truncate(3);
        s.push_str("6");
        let p_ = s.as_slice().as_ptr();
        assert_eq!(p_, p);
    }

    #[test]
    #[should_fail]
    fn test_str_truncate_invalid_len() {
        let mut s = StrBuf::from_str("12345");
        s.truncate(6);
    }

    #[test]
    #[should_fail]
    fn test_str_truncate_split_codepoint() {
        let mut s = StrBuf::from_str("\u00FC"); // ü
        s.truncate(1);
    }
}
