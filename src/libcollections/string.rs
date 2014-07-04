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

use {Collection, Mutable};
use hash;
use str;
use str::{CharRange, StrAllocating, MaybeOwned, Owned, Slice};
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
    ///
    /// # Example
    ///
    /// ```rust
    /// let hello_vec = vec![104, 101, 108, 108, 111];
    /// let string = String::from_utf8(hello_vec);
    /// assert_eq!(string, Ok("hello".to_string()));
    /// ```
    #[inline]
    pub fn from_utf8(vec: Vec<u8>) -> Result<String, Vec<u8>> {
        if str::is_utf8(vec.as_slice()) {
            Ok(String { vec: vec })
        } else {
            Err(vec)
        }
    }

    /// Converts a vector of bytes to a new utf-8 string.
    /// Any invalid utf-8 sequences are replaced with U+FFFD REPLACEMENT CHARACTER.
    ///
    /// # Example
    ///
    /// ```rust
    /// let input = b"Hello \xF0\x90\x80World";
    /// let output = String::from_utf8_lossy(input);
    /// assert_eq!(output.as_slice(), "Hello \uFFFDWorld");
    /// ```
    pub fn from_utf8_lossy<'a>(v: &'a [u8]) -> MaybeOwned<'a> {
        if str::is_utf8(v) {
            return Slice(unsafe { mem::transmute(v) })
        }

        static TAG_CONT_U8: u8 = 128u8;
        static REPLACEMENT: &'static [u8] = b"\xEF\xBF\xBD"; // U+FFFD in UTF-8
        let mut i = 0;
        let total = v.len();
        fn unsafe_get(xs: &[u8], i: uint) -> u8 {
            unsafe { *xs.unsafe_ref(i) }
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
                res.push_bytes(v.slice_to(i))
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
                        res.push_bytes(v.slice(subseqidx, i_));
                    }
                    subseqidx = i;
                    res.push_bytes(REPLACEMENT);
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
                            (0xE0        , 0xA0 .. 0xBF) => (),
                            (0xE1 .. 0xEC, 0x80 .. 0xBF) => (),
                            (0xED        , 0x80 .. 0x9F) => (),
                            (0xEE .. 0xEF, 0x80 .. 0xBF) => (),
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
                            (0xF0        , 0x90 .. 0xBF) => (),
                            (0xF1 .. 0xF3, 0x80 .. 0xBF) => (),
                            (0xF4        , 0x80 .. 0x8F) => (),
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
                res.push_bytes(v.slice(subseqidx, total))
            };
        }
        Owned(res.into_string())
    }

    /// Decode a UTF-16 encoded vector `v` into a `String`, returning `None`
    /// if `v` contains any invalid data.
    ///
    /// # Example
    ///
    /// ```rust
    /// // 𝄞music
    /// let mut v = [0xD834, 0xDD1E, 0x006d, 0x0075,
    ///              0x0073, 0x0069, 0x0063];
    /// assert_eq!(String::from_utf16(v), Some("𝄞music".to_string()));
    ///
    /// // 𝄞mu<invalid>ic
    /// v[4] = 0xD800;
    /// assert_eq!(String::from_utf16(v), None);
    /// ```
    pub fn from_utf16(v: &[u16]) -> Option<String> {
        let mut s = String::with_capacity(v.len() / 2);
        for c in str::utf16_items(v) {
            match c {
                str::ScalarValue(c) => s.push_char(c),
                str::LoneSurrogate(_) => return None
            }
        }
        Some(s)
    }

    /// Decode a UTF-16 encoded vector `v` into a string, replacing
    /// invalid data with the replacement character (U+FFFD).
    ///
    /// # Example
    /// ```rust
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
    ///          0x0073, 0xDD1E, 0x0069, 0x0063,
    ///          0xD834];
    ///
    /// assert_eq!(String::from_utf16_lossy(v),
    ///            "𝄞mus\uFFFDic\uFFFD".to_string());
    /// ```
    pub fn from_utf16_lossy(v: &[u16]) -> String {
        str::utf16_items(v).map(|c| c.to_char_lossy()).collect()
    }

    /// Convert a vector of chars to a string
    ///
    /// # Example
    ///
    /// ```rust
    /// let chars = ['h', 'e', 'l', 'l', 'o'];
    /// let string = String::from_chars(chars);
    /// assert_eq!(string.as_slice(), "hello");
    /// ```
    #[inline]
    pub fn from_chars(chs: &[char]) -> String {
        chs.iter().map(|c| *c).collect()
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

    /// Convert a byte to a UTF-8 string
    ///
    /// # Failure
    ///
    /// Fails if invalid UTF-8
    ///
    /// # Example
    ///
    /// ```rust
    /// let string = String::from_byte(104);
    /// assert_eq!(string.as_slice(), "h");
    /// ```
    pub fn from_byte(b: u8) -> String {
        assert!(b < 128u8);
        String::from_char(1, b as char)
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
        self.vec.push(byte)
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

        let byte = self.as_bytes()[len - 1];
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

impl Collection for String {
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

impl<S: Str> Add<S, String> for String {
    fn add(&self, other: &S) -> String {
        let mut s = String::from_str(self.as_slice());
        s.push_str(other.as_slice());
        return s;
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use test::Bencher;

    use Mutable;
    use str;
    use str::{Str, StrSlice, Owned, Slice};
    use super::String;
    use vec::Vec;

    #[test]
    fn test_from_str() {
      let owned: Option<::std::string::String> = from_str("string");
      assert_eq!(owned.as_ref().map(|s| s.as_slice()), Some("string"));
    }

    #[test]
    fn test_from_utf8() {
        let xs = Vec::from_slice(b"hello");
        assert_eq!(String::from_utf8(xs), Ok(String::from_str("hello")));

        let xs = Vec::from_slice("ศไทย中华Việt Nam".as_bytes());
        assert_eq!(String::from_utf8(xs), Ok(String::from_str("ศไทย中华Việt Nam")));

        let xs = Vec::from_slice(b"hello\xFF");
        assert_eq!(String::from_utf8(xs),
                   Err(Vec::from_slice(b"hello\xFF")));
    }

    #[test]
    fn test_from_utf8_lossy() {
        let xs = b"hello";
        assert_eq!(String::from_utf8_lossy(xs), Slice("hello"));

        let xs = "ศไทย中华Việt Nam".as_bytes();
        assert_eq!(String::from_utf8_lossy(xs), Slice("ศไทย中华Việt Nam"));

        let xs = b"Hello\xC2 There\xFF Goodbye";
        assert_eq!(String::from_utf8_lossy(xs),
                   Owned(String::from_str("Hello\uFFFD There\uFFFD Goodbye")));

        let xs = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
        assert_eq!(String::from_utf8_lossy(xs),
                   Owned(String::from_str("Hello\uFFFD\uFFFD There\uFFFD Goodbye")));

        let xs = b"\xF5foo\xF5\x80bar";
        assert_eq!(String::from_utf8_lossy(xs),
                   Owned(String::from_str("\uFFFDfoo\uFFFD\uFFFDbar")));

        let xs = b"\xF1foo\xF1\x80bar\xF1\x80\x80baz";
        assert_eq!(String::from_utf8_lossy(xs),
                   Owned(String::from_str("\uFFFDfoo\uFFFDbar\uFFFDbaz")));

        let xs = b"\xF4foo\xF4\x80bar\xF4\xBFbaz";
        assert_eq!(String::from_utf8_lossy(xs),
                   Owned(String::from_str("\uFFFDfoo\uFFFDbar\uFFFD\uFFFDbaz")));

        let xs = b"\xF0\x80\x80\x80foo\xF0\x90\x80\x80bar";
        assert_eq!(String::from_utf8_lossy(xs), Owned(String::from_str("\uFFFD\uFFFD\uFFFD\uFFFD\
                                               foo\U00010000bar")));

        // surrogates
        let xs = b"\xED\xA0\x80foo\xED\xBF\xBFbar";
        assert_eq!(String::from_utf8_lossy(xs), Owned(String::from_str("\uFFFD\uFFFD\uFFFDfoo\
                                               \uFFFD\uFFFD\uFFFDbar")));
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
            let s_as_utf16 = s.as_slice().utf16_units().collect::<Vec<u16>>();
            let u_as_string = String::from_utf16(u.as_slice()).unwrap();

            assert!(str::is_utf16(u.as_slice()));
            assert_eq!(s_as_utf16, u);

            assert_eq!(u_as_string, s);
            assert_eq!(String::from_utf16_lossy(u.as_slice()), s);

            assert_eq!(String::from_utf16(s_as_utf16.as_slice()).unwrap(), s);
            assert_eq!(u_as_string.as_slice().utf16_units().collect::<Vec<u16>>(), u);
        }
    }

    #[test]
    fn test_utf16_invalid() {
        // completely positive cases tested above.
        // lead + eof
        assert_eq!(String::from_utf16([0xD800]), None);
        // lead + lead
        assert_eq!(String::from_utf16([0xD800, 0xD800]), None);

        // isolated trail
        assert_eq!(String::from_utf16([0x0061, 0xDC00]), None);

        // general
        assert_eq!(String::from_utf16([0xD800, 0xd801, 0xdc8b, 0xD800]), None);
    }

    #[test]
    fn test_from_utf16_lossy() {
        // completely positive cases tested above.
        // lead + eof
        assert_eq!(String::from_utf16_lossy([0xD800]), String::from_str("\uFFFD"));
        // lead + lead
        assert_eq!(String::from_utf16_lossy([0xD800, 0xD800]), String::from_str("\uFFFD\uFFFD"));

        // isolated trail
        assert_eq!(String::from_utf16_lossy([0x0061, 0xDC00]), String::from_str("a\uFFFD"));

        // general
        assert_eq!(String::from_utf16_lossy([0xD800, 0xd801, 0xdc8b, 0xD800]),
                   String::from_str("\uFFFD𐒋\uFFFD"));
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

    #[test]
    fn test_str_add() {
        let a = String::from_str("12345");
        let b = a + "2";
        let b = b + String::from_str("2");
        assert_eq!(b.len(), 7);
        assert_eq!(b.as_slice(), "1234522");
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
        let s = "ðŒ€ðŒ–ðŒ‹ðŒ„ðŒ‘ðŒ‰à¸›à¸£Ø¯ÙˆÙ„Ø©\
            Ø§Ù„ÙƒÙˆÙŠØªà¸—à¸¨à¹„à¸—à¸¢ä¸­åŽð…ðŒ¿ðŒ»ð†ðŒ¹ðŒ»ðŒ°".as_bytes();
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
