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

use core::prelude::*;

use core::default::Default;
use core::fmt;
use core::mem;
use core::ptr;
use core::raw::Slice;

use hash;
use str;
use str::{CharRange, StrAllocating};
use vec::Vec;

/// A growable string stored as a UTF-8 encoded buffer.
#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct String {
    vec: Vec<u8>,
}

impl String {
    /// Creates a new string buffer initialized with the empty string.
    #[inline]
    pub fn new() -> String {
        String {
            vec: Vec::new(),
        }
    }

    /// Creates a new string buffer with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: uint) -> String {
        String {
            vec: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new string buffer from length, capacity, and a pointer.
    #[inline]
    pub unsafe fn from_raw_parts(length: uint, capacity: uint, ptr: *mut u8) -> String {
        String {
            vec: Vec::from_raw_parts(length, capacity, ptr),
        }
    }

    /// Creates a new string buffer from the given string.
    #[inline]
    pub fn from_str(string: &str) -> String {
        String {
            vec: Vec::from_slice(string.as_bytes())
        }
    }

    #[allow(missing_doc)]
    #[deprecated = "obsoleted by the removal of ~str"]
    #[inline]
    pub fn from_owned_str(string: String) -> String {
        string
    }

    /// Returns the vector as a string buffer, if possible, taking care not to
    /// copy it.
    ///
    /// Returns `Err` with the original vector if the vector contains invalid
    /// UTF-8.
    #[inline]
    pub fn from_utf8(vec: Vec<u8>) -> Result<String, Vec<u8>> {
        if str::is_utf8(vec.as_slice()) {
            Ok(String { vec: vec })
        } else {
            Err(vec)
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
    pub fn append(mut self, second: &str) -> String {
        self.push_str(second);
        self
    }

    /// Creates a string buffer by repeating a character `length` times.
    #[inline]
    pub fn from_char(length: uint, ch: char) -> String {
        if length == 0 {
            return String::new()
        }

        let mut buf = String::new();
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
        // This may use up to 4 bytes.
        self.vec.reserve_additional(4);

        unsafe {
            // Attempt to not use an intermediate buffer by just pushing bytes
            // directly onto this string.
            let slice = Slice {
                data: self.vec.as_ptr().offset(cur_len as int),
                len: 4,
            };
            let used = ch.encode_utf8(mem::transmute(slice));
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

    /// Works with the underlying buffer as a mutable byte slice. Unsafe
    /// because this can be used to violate the UTF-8 property.
    #[inline]
    pub unsafe fn as_mut_bytes<'a>(&'a mut self) -> &'a mut [u8] {
        self.vec.as_mut_slice()
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

    /// Removes the last character from the string buffer and returns it. Returns `None` if this
    /// string buffer is empty.
    #[inline]
    pub fn pop_char(&mut self) -> Option<char> {
        let len = self.len();
        if len == 0 {
            return None
        }

        let CharRange {ch, next} = self.as_slice().char_range_at_reverse(len);
        unsafe {
            self.vec.set_len(next);
        }
        Some(ch)
    }

    /// Removes the first byte from the string buffer and returns it. Returns `None` if this string
    /// buffer is empty.
    ///
    /// The caller must preserve the valid UTF-8 property.
    pub unsafe fn shift_byte(&mut self) -> Option<u8> {
        self.vec.shift()
    }

    /// Removes the first character from the string buffer and returns it. Returns `None` if this
    /// string buffer is empty.
    ///
    /// # Warning
    ///
    /// This is a O(n) operation as it requires copying every element in the buffer.
    pub fn shift_char (&mut self) -> Option<char> {
        let len = self.len();
        if len == 0 {
            return None
        }

        let CharRange {ch, next} = self.as_slice().char_range_at(0);
        let new_len = len - next;
        unsafe {
            ptr::copy_memory(self.vec.as_mut_ptr(), self.vec.as_ptr().offset(next as int), new_len);
            self.vec.set_len(new_len);
        }
        Some(ch)
    }

    /// Views the string buffer as a mutable sequence of bytes.
    ///
    /// Callers must preserve the valid UTF-8 property.
    pub unsafe fn as_mut_vec<'a>(&'a mut self) -> &'a mut Vec<u8> {
        &mut self.vec
    }
}

impl Container for String {
    #[inline]
    fn len(&self) -> uint {
        self.vec.len()
    }
}

impl Mutable for String {
    #[inline]
    fn clear(&mut self) {
        self.vec.clear()
    }
}

impl FromIterator<char> for String {
    fn from_iter<I:Iterator<char>>(iterator: I) -> String {
        let mut buf = String::new();
        buf.extend(iterator);
        buf
    }
}

impl Extendable<char> for String {
    fn extend<I:Iterator<char>>(&mut self, mut iterator: I) {
        for ch in iterator {
            self.push_char(ch)
        }
    }
}

impl Str for String {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str {
        unsafe {
            mem::transmute(self.vec.as_slice())
        }
    }
}

impl StrAllocating for String {
    #[inline]
    fn into_string(self) -> String {
        self
    }
}

impl Default for String {
    fn default() -> String {
        String::new()
    }
}

impl fmt::Show for String {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<H: hash::Writer> hash::Hash<H> for String {
    #[inline]
    fn hash(&self, hasher: &mut H) {
        self.as_slice().hash(hasher)
    }
}

impl<'a, S: Str> Equiv<S> for String {
    #[inline]
    fn equiv(&self, other: &S) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use test::Bencher;

    use str::{Str, StrSlice};
    use super::String;

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

    #[test]
    fn test_push_bytes() {
        let mut s = String::from_str("ABC");
        unsafe {
            s.push_bytes([ 'D' as u8 ]);
        }
        assert_eq!(s.as_slice(), "ABCD");
    }

    #[test]
    fn test_push_str() {
        let mut s = String::new();
        s.push_str("");
        assert_eq!(s.as_slice().slice_from(0), "");
        s.push_str("abc");
        assert_eq!(s.as_slice().slice_from(0), "abc");
        s.push_str("ประเทศไทย中华Việt Nam");
        assert_eq!(s.as_slice().slice_from(0), "abcประเทศไทย中华Việt Nam");
    }

    #[test]
    fn test_push_char() {
        let mut data = String::from_str("ประเทศไทย中");
        data.push_char('华');
        data.push_char('b'); // 1 byte
        data.push_char('¢'); // 2 byte
        data.push_char('€'); // 3 byte
        data.push_char('𤭢'); // 4 byte
        assert_eq!(data.as_slice(), "ประเทศไทย中华b¢€𤭢");
    }

    #[test]
    fn test_pop_char() {
        let mut data = String::from_str("ประเทศไทย中华b¢€𤭢");
        assert_eq!(data.pop_char().unwrap(), '𤭢'); // 4 bytes
        assert_eq!(data.pop_char().unwrap(), '€'); // 3 bytes
        assert_eq!(data.pop_char().unwrap(), '¢'); // 2 bytes
        assert_eq!(data.pop_char().unwrap(), 'b'); // 1 bytes
        assert_eq!(data.pop_char().unwrap(), '华');
        assert_eq!(data.as_slice(), "ประเทศไทย中");
    }

    #[test]
    fn test_shift_char() {
        let mut data = String::from_str("𤭢€¢b华ประเทศไทย中");
        assert_eq!(data.shift_char().unwrap(), '𤭢'); // 4 bytes
        assert_eq!(data.shift_char().unwrap(), '€'); // 3 bytes
        assert_eq!(data.shift_char().unwrap(), '¢'); // 2 bytes
        assert_eq!(data.shift_char().unwrap(), 'b'); // 1 bytes
        assert_eq!(data.shift_char().unwrap(), '华');
        assert_eq!(data.as_slice(), "ประเทศไทย中");
    }

    #[test]
    fn test_str_truncate() {
        let mut s = String::from_str("12345");
        s.truncate(5);
        assert_eq!(s.as_slice(), "12345");
        s.truncate(3);
        assert_eq!(s.as_slice(), "123");
        s.truncate(0);
        assert_eq!(s.as_slice(), "");

        let mut s = String::from_str("12345");
        let p = s.as_slice().as_ptr();
        s.truncate(3);
        s.push_str("6");
        let p_ = s.as_slice().as_ptr();
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
        assert_eq!(s.as_slice(), "");
    }
}
