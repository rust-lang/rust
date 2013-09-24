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
externfn!(fn puts(s: *libc::c_char))

let my_string = "Hello, world!";

// Allocate the C string with an explicit local that owns the string. The
// `c_buffer` pointer will be deallocated when `my_c_string` goes out of scope.
let my_c_string = my_string.to_c_str();
do my_c_string.with_ref |c_buffer| {
    unsafe { puts(c_buffer); }
}

// Don't save off the allocation of the C string, the `c_buffer` will be
// deallocated when this block returns!
do my_string.with_c_str |c_buffer| {
    unsafe { puts(c_buffer); }
}
 ```

*/

use cast;
use iter::{Iterator, range};
use libc;
use ops::Drop;
use option::{Option, Some, None};
use ptr::RawPtr;
use ptr;
use str;
use str::StrSlice;
use vec::{ImmutableVector, CopyableVector};
use container::Container;

/// Resolution options for the `null_byte` condition
pub enum NullByteResolution {
    /// Truncate at the null byte
    Truncate,
    /// Use a replacement byte
    ReplaceWith(libc::c_char)
}

condition! {
    // This should be &[u8] but there's a lifetime issue (#5370).
    pub null_byte: (~[u8]) -> NullByteResolution;
}

/// The representation of a C String.
///
/// This structure wraps a `*libc::c_char`, and will automatically free the
/// memory it is pointing to when it goes out of scope.
pub struct CString {
    priv buf: *libc::c_char,
    priv owns_buffer_: bool,
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
    pub fn with_ref<T>(&self, f: &fn(*libc::c_char) -> T) -> T {
        if self.buf.is_null() { fail!("CString is null!"); }
        f(self.buf)
    }

    /// Calls a closure with a mutable reference to the underlying `*libc::c_char`.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    pub fn with_mut_ref<T>(&mut self, f: &fn(*mut libc::c_char) -> T) -> T {
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
            let len = ptr::position(self.buf, |c| *c == 0);
            cast::transmute((self.buf, len + 1))
        }
    }

    /// Converts the CString into a `&str` without copying.
    /// Returns None if the CString is not UTF-8 or is null.
    #[inline]
    pub fn as_str<'a>(&'a self) -> Option<&'a str> {
        if self.buf.is_null() { return None; }
        let buf = self.as_bytes();
        let buf = buf.slice_to(buf.len()-1); // chop off the trailing NUL
        str::from_utf8_slice_opt(buf)
    }

    /// Return a CString iterator.
    pub fn iter<'a>(&'a self) -> CStringIterator<'a> {
        CStringIterator {
            ptr: self.buf,
            lifetime: unsafe { cast::transmute(self.buf) },
        }
    }
}

impl Drop for CString {
    fn drop(&mut self) {
        #[fixed_stack_segment]; #[inline(never)];
        if self.owns_buffer_ {
            unsafe {
                libc::free(self.buf as *libc::c_void)
            }
        }
    }
}

/// A generic trait for converting a value to a CString.
pub trait ToCStr {
    /// Copy the receiver into a CString.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the receiver has an interior null.
    fn to_c_str(&self) -> CString;

    /// Unsafe variant of `to_c_str()` that doesn't check for nulls.
    unsafe fn to_c_str_unchecked(&self) -> CString;

    /// Work with a temporary CString constructed from the receiver.
    /// The provided `*libc::c_char` will be freed immediately upon return.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "PATH".with_c_str(|path| libc::getenv(path))
    /// ```
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the receiver has an interior null.
    #[inline]
    fn with_c_str<T>(&self, f: &fn(*libc::c_char) -> T) -> T {
        self.to_c_str().with_ref(f)
    }

    /// Unsafe variant of `with_c_str()` that doesn't check for nulls.
    #[inline]
    unsafe fn with_c_str_unchecked<T>(&self, f: &fn(*libc::c_char) -> T) -> T {
        self.to_c_str_unchecked().with_ref(f)
    }
}

impl<'self> ToCStr for &'self str {
    #[inline]
    fn to_c_str(&self) -> CString {
        self.as_bytes().to_c_str()
    }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString {
        self.as_bytes().to_c_str_unchecked()
    }
}

impl<'self> ToCStr for &'self [u8] {
    fn to_c_str(&self) -> CString {
        #[fixed_stack_segment]; #[inline(never)];
        let mut cs = unsafe { self.to_c_str_unchecked() };
        do cs.with_mut_ref |buf| {
            for i in range(0, self.len()) {
                unsafe {
                    let p = buf.offset(i as int);
                    if *p == 0 {
                        match null_byte::cond.raise(self.to_owned()) {
                            Truncate => break,
                            ReplaceWith(c) => *p = c
                        }
                    }
                }
            }
        }
        cs
    }

    unsafe fn to_c_str_unchecked(&self) -> CString {
        #[fixed_stack_segment]; #[inline(never)];
        do self.as_imm_buf |self_buf, self_len| {
            let buf = libc::malloc(self_len as libc::size_t + 1) as *mut u8;
            if buf.is_null() {
                fail!("failed to allocate memory!");
            }

            ptr::copy_memory(buf, self_buf, self_len);
            *ptr::mut_offset(buf, self_len as int) = 0;

            CString::new(buf as *libc::c_char, true)
        }
    }
}

/// External iterator for a CString's bytes.
///
/// Use with the `std::iterator` module.
pub struct CStringIterator<'self> {
    priv ptr: *libc::c_char,
    priv lifetime: &'self libc::c_char, // FIXME: #5922
}

impl<'self> Iterator<libc::c_char> for CStringIterator<'self> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use libc;
    use ptr;
    use option::{Some, None};

    #[test]
    fn test_str_to_c_str() {
        do "".to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 0);
            }
        }

        do "hello".to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'h' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'e' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 4), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 5), 0);
            }
        }
    }

    #[test]
    fn test_vec_to_c_str() {
        let b: &[u8] = [];
        do b.to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 0);
            }
        }

        do bytes!("hello").to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'h' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'e' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 4), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 5), 0);
            }
        }

        do bytes!("foo", 0xff).to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'f' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 0xff);
                assert_eq!(*ptr::offset(buf, 4), 0);
            }
        }
    }

    #[test]
    fn test_is_null() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        assert!(c_str.is_null());
        assert!(!c_str.is_not_null());
    }

    #[test]
    fn test_unwrap() {
        #[fixed_stack_segment]; #[inline(never)];

        let c_str = "hello".to_c_str();
        unsafe { libc::free(c_str.unwrap() as *libc::c_void) }
    }

    #[test]
    fn test_with_ref() {
        #[fixed_stack_segment]; #[inline(never)];

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
        use c_str::null_byte::cond;

        let mut error_happened = false;
        do cond.trap(|err| {
            assert_eq!(err, bytes!("he", 0, "llo").to_owned())
            error_happened = true;
            Truncate
        }).inside {
            "he\x00llo".to_c_str()
        };
        assert!(error_happened);

        do cond.trap(|_| {
            ReplaceWith('?' as libc::c_char)
        }).inside(|| "he\x00llo".to_c_str()).with_ref |buf| {
            unsafe {
                assert_eq!(*buf.offset(0), 'h' as libc::c_char);
                assert_eq!(*buf.offset(1), 'e' as libc::c_char);
                assert_eq!(*buf.offset(2), '?' as libc::c_char);
                assert_eq!(*buf.offset(3), 'l' as libc::c_char);
                assert_eq!(*buf.offset(4), 'l' as libc::c_char);
                assert_eq!(*buf.offset(5), 'o' as libc::c_char);
                assert_eq!(*buf.offset(6), 0);
            }
        }
    }

    #[test]
    fn test_to_c_str_unchecked() {
        unsafe {
            do "he\x00llo".to_c_str_unchecked().with_ref |buf| {
                assert_eq!(*buf.offset(0), 'h' as libc::c_char);
                assert_eq!(*buf.offset(1), 'e' as libc::c_char);
                assert_eq!(*buf.offset(2), 0);
                assert_eq!(*buf.offset(3), 'l' as libc::c_char);
                assert_eq!(*buf.offset(4), 'l' as libc::c_char);
                assert_eq!(*buf.offset(5), 'o' as libc::c_char);
                assert_eq!(*buf.offset(6), 0);
            }
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
        let c_str = unsafe { CString::new(ptr::null(), false) };
        assert_eq!(c_str.as_str(), None);
    }
}
