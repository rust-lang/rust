// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

C-string manipulation and management

This modules provides the basic methods for creating and manipulating
null-terminated strings for use with FFI calls (back to C). Most C APIs require
that the string being passed to them is null-terminated, and by default rust's
string types are *not* null terminated.

The other problem with translating Rust strings to C strings is that Rust
strings can validly contain a null-byte in the middle of the string (0 is a
valid unicode codepoint). This means that not all Rust strings can actually be
translated to C strings.

# Creation of a C string

A C string is managed through the `CString` type defined in this module. It
"owns" the internal buffer of characters and will automatically deallocate the
buffer when the string is dropped. The `ToCStr` trait is implemented for `&str`
and `&[u8]`, but the conversions can fail due to some of the limitations
explained above.

This also means that currently whenever a C string is created, an allocation
must be performed to place the data elsewhere (the lifetime of the C string is
not tied to the lifetime of the original string/data buffer). If C strings are
heavily used in applications, then caching may be advisable to prevent
unnecessary amounts of allocations.

An example of creating and using a C string would be:

```rust
use std::libc;
extern {
    fn puts(s: *libc::c_char);
}

let my_string = "Hello, world!";

// Allocate the C string with an explicit local that owns the string. The
// `c_buffer` pointer will be deallocated when `my_c_string` goes out of scope.
let my_c_string = my_string.to_c_str();
my_c_string.with_ref(|c_buffer| {
    unsafe { puts(c_buffer); }
});

// Don't save off the allocation of the C string, the `c_buffer` will be
// deallocated when this block returns!
my_string.with_c_str(|c_buffer| {
    unsafe { puts(c_buffer); }
});
 ```

*/

use cast;
use container::Container;
use iter::{Iterator, range};
use libc;
use kinds::marker;
use ops::Drop;
use cmp::Eq;
use clone::Clone;
use option::{Option, Some, None};
use ptr::RawPtr;
use ptr;
use str::StrSlice;
use str;
use vec::{ImmutableVector, MutableVector};
use vec;
use unstable::intrinsics;
use rt::global_heap::malloc_raw;

/// The representation of a C String.
///
/// This structure wraps a `*libc::c_char`, and will automatically free the
/// memory it is pointing to when it goes out of scope.
pub struct CString {
    priv buf: *libc::c_char,
    priv owns_buffer_: bool,
}

impl Clone for CString {
    /// Clone this CString into a new, uniquely owned CString. For safety
    /// reasons, this is always a deep clone, rather than the usual shallow
    /// clone.
    fn clone(&self) -> CString {
        if self.buf.is_null() {
            CString { buf: self.buf, owns_buffer_: self.owns_buffer_ }
        } else {
            let len = self.len() + 1;
            let buf = unsafe { malloc_raw(len) } as *mut libc::c_char;
            unsafe { ptr::copy_nonoverlapping_memory(buf, self.buf, len); }
            CString { buf: buf as *libc::c_char, owns_buffer_: true }
        }
    }
}

impl Eq for CString {
    fn eq(&self, other: &CString) -> bool {
        if self.buf as uint == other.buf as uint {
            true
        } else if self.buf.is_null() || other.buf.is_null() {
            false
        } else {
            unsafe {
                libc::strcmp(self.buf, other.buf) == 0
            }
        }
    }
}

impl CString {
    /// Create a C String from a pointer.
    pub unsafe fn new(buf: *libc::c_char, owns_buffer: bool) -> CString {
        CString { buf: buf, owns_buffer_: owns_buffer }
    }

    /// Unwraps the wrapped `*libc::c_char` from the `CString` wrapper.
    /// Any ownership of the buffer by the `CString` wrapper is forgotten.
    pub unsafe fn unwrap(self) -> *libc::c_char {
        let mut c_str = self;
        c_str.owns_buffer_ = false;
        c_str.buf
    }

    /// Calls a closure with a reference to the underlying `*libc::c_char`.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    pub fn with_ref<T>(&self, f: |*libc::c_char| -> T) -> T {
        if self.buf.is_null() { fail!("CString is null!"); }
        f(self.buf)
    }

    /// Calls a closure with a mutable reference to the underlying `*libc::c_char`.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    pub fn with_mut_ref<T>(&mut self, f: |*mut libc::c_char| -> T) -> T {
        if self.buf.is_null() { fail!("CString is null!"); }
        f(unsafe { cast::transmute_mut_unsafe(self.buf) })
    }

    /// Returns true if the CString is a null.
    pub fn is_null(&self) -> bool {
        self.buf.is_null()
    }

    /// Returns true if the CString is not null.
    pub fn is_not_null(&self) -> bool {
        self.buf.is_not_null()
    }

    /// Returns whether or not the `CString` owns the buffer.
    pub fn owns_buffer(&self) -> bool {
        self.owns_buffer_
    }

    /// Converts the CString into a `&[u8]` without copying.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[inline]
    pub fn as_bytes<'a>(&'a self) -> &'a [u8] {
        if self.buf.is_null() { fail!("CString is null!"); }
        unsafe {
            cast::transmute((self.buf, self.len() + 1))
        }
    }

    /// Converts the CString into a `&str` without copying.
    /// Returns None if the CString is not UTF-8.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[inline]
    pub fn as_str<'a>(&'a self) -> Option<&'a str> {
        let buf = self.as_bytes();
        let buf = buf.slice_to(buf.len()-1); // chop off the trailing NUL
        str::from_utf8(buf)
    }

    /// Return a CString iterator.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    pub fn iter<'a>(&'a self) -> CChars<'a> {
        if self.buf.is_null() { fail!("CString is null!"); }
        CChars {
            ptr: self.buf,
            marker: marker::ContravariantLifetime,
        }
    }
}

impl Drop for CString {
    fn drop(&mut self) {
        if self.owns_buffer_ {
            unsafe {
                libc::free(self.buf as *mut libc::c_void)
            }
        }
    }
}

impl Container for CString {
    /// Return the number of bytes in the CString (not including the NUL terminator).
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[inline]
    fn len(&self) -> uint {
        if self.buf.is_null() { fail!("CString is null!"); }
        unsafe {
            ptr::position(self.buf, |c| *c == 0)
        }
    }
}

/// A generic trait for converting a value to a CString.
pub trait ToCStr {
    /// Copy the receiver into a CString.
    ///
    /// # Failure
    ///
    /// Fails the task if the receiver has an interior null.
    fn to_c_str(&self) -> CString;

    /// Unsafe variant of `to_c_str()` that doesn't check for nulls.
    unsafe fn to_c_str_unchecked(&self) -> CString;

    /// Work with a temporary CString constructed from the receiver.
    /// The provided `*libc::c_char` will be freed immediately upon return.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::libc;
    ///
    /// let s = "PATH".with_c_str(|path| unsafe {
    ///     libc::getenv(path)
    /// });
    /// ```
    ///
    /// # Failure
    ///
    /// Fails the task if the receiver has an interior null.
    #[inline]
    fn with_c_str<T>(&self, f: |*libc::c_char| -> T) -> T {
        self.to_c_str().with_ref(f)
    }

    /// Unsafe variant of `with_c_str()` that doesn't check for nulls.
    #[inline]
    unsafe fn with_c_str_unchecked<T>(&self, f: |*libc::c_char| -> T) -> T {
        self.to_c_str_unchecked().with_ref(f)
    }
}

impl<'a> ToCStr for &'a str {
    #[inline]
    fn to_c_str(&self) -> CString {
        self.as_bytes().to_c_str()
    }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString {
        self.as_bytes().to_c_str_unchecked()
    }

    #[inline]
    fn with_c_str<T>(&self, f: |*libc::c_char| -> T) -> T {
        self.as_bytes().with_c_str(f)
    }

    #[inline]
    unsafe fn with_c_str_unchecked<T>(&self, f: |*libc::c_char| -> T) -> T {
        self.as_bytes().with_c_str_unchecked(f)
    }
}

// The length of the stack allocated buffer for `vec.with_c_str()`
static BUF_LEN: uint = 128;

impl<'a> ToCStr for &'a [u8] {
    fn to_c_str(&self) -> CString {
        let mut cs = unsafe { self.to_c_str_unchecked() };
        cs.with_mut_ref(|buf| check_for_null(*self, buf));
        cs
    }

    unsafe fn to_c_str_unchecked(&self) -> CString {
        let self_len = self.len();
        let buf = malloc_raw(self_len + 1);

        ptr::copy_memory(buf, self.as_ptr(), self_len);
        *ptr::mut_offset(buf, self_len as int) = 0;

        CString::new(buf as *libc::c_char, true)
    }

    fn with_c_str<T>(&self, f: |*libc::c_char| -> T) -> T {
        unsafe { with_c_str(*self, true, f) }
    }

    unsafe fn with_c_str_unchecked<T>(&self, f: |*libc::c_char| -> T) -> T {
        with_c_str(*self, false, f)
    }
}

// Unsafe function that handles possibly copying the &[u8] into a stack array.
unsafe fn with_c_str<T>(v: &[u8], checked: bool, f: |*libc::c_char| -> T) -> T {
    if v.len() < BUF_LEN {
        let mut buf: [u8, .. BUF_LEN] = intrinsics::uninit();
        vec::bytes::copy_memory(buf, v);
        buf[v.len()] = 0;

        let buf = buf.as_mut_ptr();
        if checked {
            check_for_null(v, buf as *mut libc::c_char);
        }

        f(buf as *libc::c_char)
    } else if checked {
        v.to_c_str().with_ref(f)
    } else {
        v.to_c_str_unchecked().with_ref(f)
    }
}

#[inline]
fn check_for_null(v: &[u8], buf: *mut libc::c_char) {
    for i in range(0, v.len()) {
        unsafe {
            let p = buf.offset(i as int);
            assert!(*p != 0);
        }
    }
}

/// External iterator for a CString's bytes.
///
/// Use with the `std::iter` module.
pub struct CChars<'a> {
    priv ptr: *libc::c_char,
    priv marker: marker::ContravariantLifetime<'a>,
}

impl<'a> Iterator<libc::c_char> for CChars<'a> {
    fn next(&mut self) -> Option<libc::c_char> {
        let ch = unsafe { *self.ptr };
        if ch == 0 {
            None
        } else {
            self.ptr = unsafe { ptr::offset(self.ptr, 1) };
            Some(ch)
        }
    }
}

/// Parses a C "multistring", eg windows env values or
/// the req->ptr result in a uv_fs_readdir() call.
///
/// Optionally, a `count` can be passed in, limiting the
/// parsing to only being done `count`-times.
///
/// The specified closure is invoked with each string that
/// is found, and the number of strings found is returned.
pub unsafe fn from_c_multistring(buf: *libc::c_char,
                                 count: Option<uint>,
                                 f: |&CString|) -> uint {

    let mut curr_ptr: uint = buf as uint;
    let mut ctr = 0;
    let (limited_count, limit) = match count {
        Some(limit) => (true, limit),
        None => (false, 0)
    };
    while ((limited_count && ctr < limit) || !limited_count)
          && *(curr_ptr as *libc::c_char) != 0 as libc::c_char {
        let cstr = CString::new(curr_ptr as *libc::c_char, false);
        f(&cstr);
        curr_ptr += cstr.len() + 1;
        ctr += 1;
    }
    return ctr;
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use libc;
    use ptr;

    #[test]
    fn test_str_multistring_parsing() {
        unsafe {
            let input = bytes!("zero", "\x00", "one", "\x00", "\x00");
            let ptr = input.as_ptr();
            let expected = ["zero", "one"];
            let mut it = expected.iter();
            let result = from_c_multistring(ptr as *libc::c_char, None, |c| {
                let cbytes = c.as_bytes().slice_to(c.len());
                assert_eq!(cbytes, it.next().unwrap().as_bytes());
            });
            assert_eq!(result, 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_str_to_c_str() {
        "".to_c_str().with_ref(|buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 0);
            }
        });

        "hello".to_c_str().with_ref(|buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'h' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'e' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 4), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 5), 0);
            }
        })
    }

    #[test]
    fn test_vec_to_c_str() {
        let b: &[u8] = [];
        b.to_c_str().with_ref(|buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 0);
            }
        });

        let _ = bytes!("hello").to_c_str().with_ref(|buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'h' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'e' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 4), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 5), 0);
            }
        });

        let _ = bytes!("foo", 0xff).to_c_str().with_ref(|buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'f' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 0xff as i8);
                assert_eq!(*ptr::offset(buf, 4), 0);
            }
        });
    }

    #[test]
    fn test_is_null() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        assert!(c_str.is_null());
        assert!(!c_str.is_not_null());
    }

    #[test]
    fn test_unwrap() {
        let c_str = "hello".to_c_str();
        unsafe { libc::free(c_str.unwrap() as *mut libc::c_void) }
    }

    #[test]
    fn test_with_ref() {
        let c_str = "hello".to_c_str();
        let len = unsafe { c_str.with_ref(|buf| libc::strlen(buf)) };
        assert!(!c_str.is_null());
        assert!(c_str.is_not_null());
        assert_eq!(len, 5);
    }

    #[test]
    #[should_fail]
    fn test_with_ref_empty_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.with_ref(|_| ());
    }

    #[test]
    fn test_iterator() {
        let c_str = "".to_c_str();
        let mut iter = c_str.iter();
        assert_eq!(iter.next(), None);

        let c_str = "hello".to_c_str();
        let mut iter = c_str.iter();
        assert_eq!(iter.next(), Some('h' as libc::c_char));
        assert_eq!(iter.next(), Some('e' as libc::c_char));
        assert_eq!(iter.next(), Some('l' as libc::c_char));
        assert_eq!(iter.next(), Some('l' as libc::c_char));
        assert_eq!(iter.next(), Some('o' as libc::c_char));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_to_c_str_fail() {
        use task;
        assert!(task::try(proc() { "he\x00llo".to_c_str() }).is_err());
    }

    #[test]
    fn test_to_c_str_unchecked() {
        unsafe {
            "he\x00llo".to_c_str_unchecked().with_ref(|buf| {
                assert_eq!(*buf.offset(0), 'h' as libc::c_char);
                assert_eq!(*buf.offset(1), 'e' as libc::c_char);
                assert_eq!(*buf.offset(2), 0);
                assert_eq!(*buf.offset(3), 'l' as libc::c_char);
                assert_eq!(*buf.offset(4), 'l' as libc::c_char);
                assert_eq!(*buf.offset(5), 'o' as libc::c_char);
                assert_eq!(*buf.offset(6), 0);
            })
        }
    }

    #[test]
    fn test_as_bytes() {
        let c_str = "hello".to_c_str();
        assert_eq!(c_str.as_bytes(), bytes!("hello", 0));
        let c_str = "".to_c_str();
        assert_eq!(c_str.as_bytes(), bytes!(0));
        let c_str = bytes!("foo", 0xff).to_c_str();
        assert_eq!(c_str.as_bytes(), bytes!("foo", 0xff, 0));
    }

    #[test]
    #[should_fail]
    fn test_as_bytes_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.as_bytes();
    }

    #[test]
    fn test_as_str() {
        let c_str = "hello".to_c_str();
        assert_eq!(c_str.as_str(), Some("hello"));
        let c_str = "".to_c_str();
        assert_eq!(c_str.as_str(), Some(""));
        let c_str = bytes!("foo", 0xff).to_c_str();
        assert_eq!(c_str.as_str(), None);
    }

    #[test]
    #[should_fail]
    fn test_as_str_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.as_str();
    }

    #[test]
    #[should_fail]
    fn test_len_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.len();
    }

    #[test]
    #[should_fail]
    fn test_iter_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.iter();
    }

    #[test]
    fn test_clone() {
        let a = "hello".to_c_str();
        let b = a.clone();
        assert!(a == b);
    }

    #[test]
    fn test_clone_noleak() {
        fn foo(f: |c: &CString|) {
            let s = ~"test";
            let c = s.to_c_str();
            // give the closure a non-owned CString
            let mut c_ = c.with_ref(|c| unsafe { CString::new(c, false) } );
            f(&c_);
            // muck with the buffer for later printing
            c_.with_mut_ref(|c| unsafe { *c = 'X' as libc::c_char } );
        }

        let mut c_: Option<CString> = None;
        foo(|c| {
            c_ = Some(c.clone());
            c.clone();
            // force a copy, reading the memory
            c.as_bytes().to_owned();
        });
        let c_ = c_.unwrap();
        // force a copy, reading the memory
        c_.as_bytes().to_owned();
    }

    #[test]
    fn test_clone_eq_null() {
        let x = unsafe { CString::new(ptr::null(), false) };
        let y = x.clone();
        assert!(x == y);
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use libc;
    use prelude::*;
    use ptr;

    #[inline]
    fn check(s: &str, c_str: *libc::c_char) {
        let s_buf = s.as_ptr();
        for i in range(0, s.len()) {
            unsafe {
                assert_eq!(
                    *ptr::offset(s_buf, i as int) as libc::c_char,
                    *ptr::offset(c_str, i as int));
            }
        }
    }

    static s_short: &'static str = "Mary";
    static s_medium: &'static str = "Mary had a little lamb";
    static s_long: &'static str = "\
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb";

    fn bench_to_str(bh: &mut BenchHarness, s: &str) {
        bh.iter(|| {
            let c_str = s.to_c_str();
            c_str.with_ref(|c_str_buf| check(s, c_str_buf))
        })
    }

    #[bench]
    fn bench_to_c_str_short(bh: &mut BenchHarness) {
        bench_to_str(bh, s_short)
    }

    #[bench]
    fn bench_to_c_str_medium(bh: &mut BenchHarness) {
        bench_to_str(bh, s_medium)
    }

    #[bench]
    fn bench_to_c_str_long(bh: &mut BenchHarness) {
        bench_to_str(bh, s_long)
    }

    fn bench_to_c_str_unchecked(bh: &mut BenchHarness, s: &str) {
        bh.iter(|| {
            let c_str = unsafe { s.to_c_str_unchecked() };
            c_str.with_ref(|c_str_buf| check(s, c_str_buf))
        })
    }

    #[bench]
    fn bench_to_c_str_unchecked_short(bh: &mut BenchHarness) {
        bench_to_c_str_unchecked(bh, s_short)
    }

    #[bench]
    fn bench_to_c_str_unchecked_medium(bh: &mut BenchHarness) {
        bench_to_c_str_unchecked(bh, s_medium)
    }

    #[bench]
    fn bench_to_c_str_unchecked_long(bh: &mut BenchHarness) {
        bench_to_c_str_unchecked(bh, s_long)
    }

    fn bench_with_c_str(bh: &mut BenchHarness, s: &str) {
        bh.iter(|| {
            s.with_c_str(|c_str_buf| check(s, c_str_buf))
        })
    }

    #[bench]
    fn bench_with_c_str_short(bh: &mut BenchHarness) {
        bench_with_c_str(bh, s_short)
    }

    #[bench]
    fn bench_with_c_str_medium(bh: &mut BenchHarness) {
        bench_with_c_str(bh, s_medium)
    }

    #[bench]
    fn bench_with_c_str_long(bh: &mut BenchHarness) {
        bench_with_c_str(bh, s_long)
    }

    fn bench_with_c_str_unchecked(bh: &mut BenchHarness, s: &str) {
        bh.iter(|| {
            unsafe {
                s.with_c_str_unchecked(|c_str_buf| check(s, c_str_buf))
            }
        })
    }

    #[bench]
    fn bench_with_c_str_unchecked_short(bh: &mut BenchHarness) {
        bench_with_c_str_unchecked(bh, s_short)
    }

    #[bench]
    fn bench_with_c_str_unchecked_medium(bh: &mut BenchHarness) {
        bench_with_c_str_unchecked(bh, s_medium)
    }

    #[bench]
    fn bench_with_c_str_unchecked_long(bh: &mut BenchHarness) {
        bench_with_c_str_unchecked(bh, s_long)
    }
}
