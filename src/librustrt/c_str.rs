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
extern crate libc;

extern {
    fn puts(s: *const libc::c_char);
}

fn main() {
    let my_string = "Hello, world!";

    // Allocate the C string with an explicit local that owns the string. The
    // `c_buffer` pointer will be deallocated when `my_c_string` goes out of scope.
    let my_c_string = my_string.to_c_str();
    unsafe {
        puts(my_c_string.as_ptr());
    }

    // Don't save/return the pointer to the C string, the `c_buffer` will be
    // deallocated when this block returns!
    my_string.with_c_str(|c_buffer| {
        unsafe { puts(c_buffer); }
    });
}
```

*/

use core::prelude::*;

use alloc::libc_heap::malloc_raw;
use collections::string::String;
use collections::hash;
use core::kinds::marker;
use core::mem;
use core::ptr;
use core::raw::Slice;
use core::slice;
use core::str;
use libc;

/// The representation of a C String.
///
/// This structure wraps a `*libc::c_char`, and will automatically free the
/// memory it is pointing to when it goes out of scope.
pub struct CString {
    buf: *const libc::c_char,
    owns_buffer_: bool,
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
            CString { buf: buf as *const libc::c_char, owns_buffer_: true }
        }
    }
}

impl PartialEq for CString {
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

impl PartialOrd for CString {
    #[inline]
    fn partial_cmp(&self, other: &CString) -> Option<Ordering> {
        self.as_bytes().partial_cmp(&other.as_bytes())
    }
}

impl Eq for CString {}

impl<S: hash::Writer> hash::Hash<S> for CString {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.as_bytes().hash(state)
    }
}

impl CString {
    /// Create a C String from a pointer.
    pub unsafe fn new(buf: *const libc::c_char, owns_buffer: bool) -> CString {
        CString { buf: buf, owns_buffer_: owns_buffer }
    }

    /// Return a pointer to the NUL-terminated string data.
    ///
    /// `.as_ptr` returns an internal pointer into the `CString`, and
    /// may be invalidated when the `CString` falls out of scope (the
    /// destructor will run, freeing the allocation if there is
    /// one).
    ///
    /// ```rust
    /// let foo = "some string";
    ///
    /// // right
    /// let x = foo.to_c_str();
    /// let p = x.as_ptr();
    ///
    /// // wrong (the CString will be freed, invalidating `p`)
    /// let p = foo.to_c_str().as_ptr();
    /// ```
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    ///
    /// # Example
    ///
    /// ```rust
    /// extern crate libc;
    ///
    /// fn main() {
    ///     let c_str = "foo bar".to_c_str();
    ///     unsafe {
    ///         libc::puts(c_str.as_ptr());
    ///     }
    /// }
    /// ```
    pub fn as_ptr(&self) -> *const libc::c_char {
        if self.buf.is_null() { fail!("CString is null!"); }

        self.buf
    }

    /// Return a mutable pointer to the NUL-terminated string data.
    ///
    /// `.as_mut_ptr` returns an internal pointer into the `CString`, and
    /// may be invalidated when the `CString` falls out of scope (the
    /// destructor will run, freeing the allocation if there is
    /// one).
    ///
    /// ```rust
    /// let foo = "some string";
    ///
    /// // right
    /// let mut x = foo.to_c_str();
    /// let p = x.as_mut_ptr();
    ///
    /// // wrong (the CString will be freed, invalidating `p`)
    /// let p = foo.to_c_str().as_mut_ptr();
    /// ```
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    pub fn as_mut_ptr(&mut self) -> *mut libc::c_char {
        if self.buf.is_null() { fail!("CString is null!") }

        self.buf as *mut _
    }

    /// Calls a closure with a reference to the underlying `*libc::c_char`.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[deprecated="use `.as_ptr()`"]
    pub fn with_ref<T>(&self, f: |*const libc::c_char| -> T) -> T {
        if self.buf.is_null() { fail!("CString is null!"); }
        f(self.buf)
    }

    /// Calls a closure with a mutable reference to the underlying `*libc::c_char`.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[deprecated="use `.as_mut_ptr()`"]
    pub fn with_mut_ref<T>(&mut self, f: |*mut libc::c_char| -> T) -> T {
        if self.buf.is_null() { fail!("CString is null!"); }
        f(self.buf as *mut libc::c_char)
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
    /// Includes the terminating NUL byte.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[inline]
    pub fn as_bytes<'a>(&'a self) -> &'a [u8] {
        if self.buf.is_null() { fail!("CString is null!"); }
        unsafe {
            mem::transmute(Slice { data: self.buf, len: self.len() + 1 })
        }
    }

    /// Converts the CString into a `&[u8]` without copying.
    /// Does not include the terminating NUL byte.
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[inline]
    pub fn as_bytes_no_nul<'a>(&'a self) -> &'a [u8] {
        if self.buf.is_null() { fail!("CString is null!"); }
        unsafe {
            mem::transmute(Slice { data: self.buf, len: self.len() })
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
        let buf = self.as_bytes_no_nul();
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

    /// Unwraps the wrapped `*libc::c_char` from the `CString` wrapper.
    ///
    /// Any ownership of the buffer by the `CString` wrapper is
    /// forgotten, meaning that the backing allocation of this
    /// `CString` is not automatically freed if it owns the
    /// allocation. In this case, a user of `.unwrap()` should ensure
    /// the allocation is freed, to avoid leaking memory.
    ///
    /// Prefer `.as_ptr()` when just retrieving a pointer to the
    /// string data, as that does not relinquish ownership.
    pub unsafe fn unwrap(mut self) -> *const libc::c_char {
        self.owns_buffer_ = false;
        self.buf
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

impl Collection for CString {
    /// Return the number of bytes in the CString (not including the NUL terminator).
    ///
    /// # Failure
    ///
    /// Fails if the CString is null.
    #[inline]
    fn len(&self) -> uint {
        if self.buf.is_null() { fail!("CString is null!"); }
        let mut cur = self.buf;
        let mut len = 0;
        unsafe {
            while *cur != 0 {
                len += 1;
                cur = cur.offset(1);
            }
        }
        return len;
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
    /// extern crate libc;
    ///
    /// fn main() {
    ///     let s = "PATH".with_c_str(|path| unsafe {
    ///         libc::getenv(path)
    ///     });
    /// }
    /// ```
    ///
    /// # Failure
    ///
    /// Fails the task if the receiver has an interior null.
    #[inline]
    fn with_c_str<T>(&self, f: |*const libc::c_char| -> T) -> T {
        let c_str = self.to_c_str();
        f(c_str.as_ptr())
    }

    /// Unsafe variant of `with_c_str()` that doesn't check for nulls.
    #[inline]
    unsafe fn with_c_str_unchecked<T>(&self, f: |*const libc::c_char| -> T) -> T {
        let c_str = self.to_c_str_unchecked();
        f(c_str.as_ptr())
    }
}

// FIXME (#12938): Until DST lands, we cannot decompose &str into &
// and str, so we cannot usefully take ToCStr arguments by reference
// (without forcing an additional & around &str). So we are instead
// temporarily adding an instance for ~str and String, so that we can
// take ToCStr as owned. When DST lands, the string instances should
// be revisited, and arguments bound by ToCStr should be passed by
// reference.

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
    fn with_c_str<T>(&self, f: |*const libc::c_char| -> T) -> T {
        self.as_bytes().with_c_str(f)
    }

    #[inline]
    unsafe fn with_c_str_unchecked<T>(&self, f: |*const libc::c_char| -> T) -> T {
        self.as_bytes().with_c_str_unchecked(f)
    }
}

impl ToCStr for String {
    #[inline]
    fn to_c_str(&self) -> CString {
        self.as_bytes().to_c_str()
    }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString {
        self.as_bytes().to_c_str_unchecked()
    }

    #[inline]
    fn with_c_str<T>(&self, f: |*const libc::c_char| -> T) -> T {
        self.as_bytes().with_c_str(f)
    }

    #[inline]
    unsafe fn with_c_str_unchecked<T>(&self, f: |*const libc::c_char| -> T) -> T {
        self.as_bytes().with_c_str_unchecked(f)
    }
}

// The length of the stack allocated buffer for `vec.with_c_str()`
static BUF_LEN: uint = 128;

impl<'a> ToCStr for &'a [u8] {
    fn to_c_str(&self) -> CString {
        let mut cs = unsafe { self.to_c_str_unchecked() };
        check_for_null(*self, cs.as_mut_ptr());
        cs
    }

    unsafe fn to_c_str_unchecked(&self) -> CString {
        let self_len = self.len();
        let buf = malloc_raw(self_len + 1);

        ptr::copy_memory(buf, self.as_ptr(), self_len);
        *buf.offset(self_len as int) = 0;

        CString::new(buf as *const libc::c_char, true)
    }

    fn with_c_str<T>(&self, f: |*const libc::c_char| -> T) -> T {
        unsafe { with_c_str(*self, true, f) }
    }

    unsafe fn with_c_str_unchecked<T>(&self, f: |*const libc::c_char| -> T) -> T {
        with_c_str(*self, false, f)
    }
}

// Unsafe function that handles possibly copying the &[u8] into a stack array.
unsafe fn with_c_str<T>(v: &[u8], checked: bool,
                        f: |*const libc::c_char| -> T) -> T {
    let c_str = if v.len() < BUF_LEN {
        let mut buf: [u8, .. BUF_LEN] = mem::uninitialized();
        slice::bytes::copy_memory(buf, v);
        buf[v.len()] = 0;

        let buf = buf.as_mut_ptr();
        if checked {
            check_for_null(v, buf as *mut libc::c_char);
        }

        return f(buf as *const libc::c_char)
    } else if checked {
        v.to_c_str()
    } else {
        v.to_c_str_unchecked()
    };

    f(c_str.as_ptr())
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
    ptr: *const libc::c_char,
    marker: marker::ContravariantLifetime<'a>,
}

impl<'a> Iterator<libc::c_char> for CChars<'a> {
    fn next(&mut self) -> Option<libc::c_char> {
        let ch = unsafe { *self.ptr };
        if ch == 0 {
            None
        } else {
            self.ptr = unsafe { self.ptr.offset(1) };
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
pub unsafe fn from_c_multistring(buf: *const libc::c_char,
                                 count: Option<uint>,
                                 f: |&CString|) -> uint {

    let mut curr_ptr: uint = buf as uint;
    let mut ctr = 0;
    let (limited_count, limit) = match count {
        Some(limit) => (true, limit),
        None => (false, 0)
    };
    while ((limited_count && ctr < limit) || !limited_count)
          && *(curr_ptr as *const libc::c_char) != 0 as libc::c_char {
        let cstr = CString::new(curr_ptr as *const libc::c_char, false);
        f(&cstr);
        curr_ptr += cstr.len() + 1;
        ctr += 1;
    }
    return ctr;
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use std::ptr;
    use std::task;
    use libc;

    use super::*;

    #[test]
    fn test_str_multistring_parsing() {
        unsafe {
            let input = b"zero\0one\0\0";
            let ptr = input.as_ptr();
            let expected = ["zero", "one"];
            let mut it = expected.iter();
            let result = from_c_multistring(ptr as *const libc::c_char, None, |c| {
                let cbytes = c.as_bytes_no_nul();
                assert_eq!(cbytes, it.next().unwrap().as_bytes());
            });
            assert_eq!(result, 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_str_to_c_str() {
        let c_str = "".to_c_str();
        unsafe {
            assert_eq!(*c_str.as_ptr().offset(0), 0);
        }

        let c_str = "hello".to_c_str();
        let buf = c_str.as_ptr();
        unsafe {
            assert_eq!(*buf.offset(0), 'h' as libc::c_char);
            assert_eq!(*buf.offset(1), 'e' as libc::c_char);
            assert_eq!(*buf.offset(2), 'l' as libc::c_char);
            assert_eq!(*buf.offset(3), 'l' as libc::c_char);
            assert_eq!(*buf.offset(4), 'o' as libc::c_char);
            assert_eq!(*buf.offset(5), 0);
        }
    }

    #[test]
    fn test_vec_to_c_str() {
        let b: &[u8] = [];
        let c_str = b.to_c_str();
        unsafe {
            assert_eq!(*c_str.as_ptr().offset(0), 0);
        }

        let c_str = b"hello".to_c_str();
        let buf = c_str.as_ptr();
        unsafe {
            assert_eq!(*buf.offset(0), 'h' as libc::c_char);
            assert_eq!(*buf.offset(1), 'e' as libc::c_char);
            assert_eq!(*buf.offset(2), 'l' as libc::c_char);
            assert_eq!(*buf.offset(3), 'l' as libc::c_char);
            assert_eq!(*buf.offset(4), 'o' as libc::c_char);
            assert_eq!(*buf.offset(5), 0);
        }

        let c_str = b"foo\xFF".to_c_str();
        let buf = c_str.as_ptr();
        unsafe {
            assert_eq!(*buf.offset(0), 'f' as libc::c_char);
            assert_eq!(*buf.offset(1), 'o' as libc::c_char);
            assert_eq!(*buf.offset(2), 'o' as libc::c_char);
            assert_eq!(*buf.offset(3), 0xffu8 as i8);
            assert_eq!(*buf.offset(4), 0);
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
        let c_str = "hello".to_c_str();
        unsafe { libc::free(c_str.unwrap() as *mut libc::c_void) }
    }

    #[test]
    fn test_as_ptr() {
        let c_str = "hello".to_c_str();
        let len = unsafe { libc::strlen(c_str.as_ptr()) };
        assert!(!c_str.is_null());
        assert!(c_str.is_not_null());
        assert_eq!(len, 5);
    }
    #[test]
    #[should_fail]
    fn test_as_ptr_empty_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.as_ptr();
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
        assert!(task::try(proc() { "he\x00llo".to_c_str() }).is_err());
    }

    #[test]
    fn test_to_c_str_unchecked() {
        unsafe {
            let c_string = "he\x00llo".to_c_str_unchecked();
            let buf = c_string.as_ptr();
            assert_eq!(*buf.offset(0), 'h' as libc::c_char);
            assert_eq!(*buf.offset(1), 'e' as libc::c_char);
            assert_eq!(*buf.offset(2), 0);
            assert_eq!(*buf.offset(3), 'l' as libc::c_char);
            assert_eq!(*buf.offset(4), 'l' as libc::c_char);
            assert_eq!(*buf.offset(5), 'o' as libc::c_char);
            assert_eq!(*buf.offset(6), 0);
        }
    }

    #[test]
    fn test_as_bytes() {
        let c_str = "hello".to_c_str();
        assert_eq!(c_str.as_bytes(), b"hello\0");
        let c_str = "".to_c_str();
        assert_eq!(c_str.as_bytes(), b"\0");
        let c_str = b"foo\xFF".to_c_str();
        assert_eq!(c_str.as_bytes(), b"foo\xFF\0");
    }

    #[test]
    fn test_as_bytes_no_nul() {
        let c_str = "hello".to_c_str();
        assert_eq!(c_str.as_bytes_no_nul(), b"hello");
        let c_str = "".to_c_str();
        let exp: &[u8] = [];
        assert_eq!(c_str.as_bytes_no_nul(), exp);
        let c_str = b"foo\xFF".to_c_str();
        assert_eq!(c_str.as_bytes_no_nul(), b"foo\xFF");
    }

    #[test]
    #[should_fail]
    fn test_as_bytes_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.as_bytes();
    }

    #[test]
    #[should_fail]
    fn test_as_bytes_no_nul_fail() {
        let c_str = unsafe { CString::new(ptr::null(), false) };
        c_str.as_bytes_no_nul();
    }

    #[test]
    fn test_as_str() {
        let c_str = "hello".to_c_str();
        assert_eq!(c_str.as_str(), Some("hello"));
        let c_str = "".to_c_str();
        assert_eq!(c_str.as_str(), Some(""));
        let c_str = b"foo\xFF".to_c_str();
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
            let s = "test".to_string();
            let c = s.to_c_str();
            // give the closure a non-owned CString
            let mut c_ = unsafe { CString::new(c.as_ptr(), false) };
            f(&c_);
            // muck with the buffer for later printing
            unsafe { *c_.as_mut_ptr() = 'X' as libc::c_char }
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
    use test::Bencher;
    use libc;
    use std::prelude::*;

    #[inline]
    fn check(s: &str, c_str: *const libc::c_char) {
        let s_buf = s.as_ptr();
        for i in range(0, s.len()) {
            unsafe {
                assert_eq!(
                    *s_buf.offset(i as int) as libc::c_char,
                    *c_str.offset(i as int));
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

    fn bench_to_string(b: &mut Bencher, s: &str) {
        b.iter(|| {
            let c_str = s.to_c_str();
            check(s, c_str.as_ptr());
        })
    }

    #[bench]
    fn bench_to_c_str_short(b: &mut Bencher) {
        bench_to_string(b, s_short)
    }

    #[bench]
    fn bench_to_c_str_medium(b: &mut Bencher) {
        bench_to_string(b, s_medium)
    }

    #[bench]
    fn bench_to_c_str_long(b: &mut Bencher) {
        bench_to_string(b, s_long)
    }

    fn bench_to_c_str_unchecked(b: &mut Bencher, s: &str) {
        b.iter(|| {
            let c_str = unsafe { s.to_c_str_unchecked() };
            check(s, c_str.as_ptr())
        })
    }

    #[bench]
    fn bench_to_c_str_unchecked_short(b: &mut Bencher) {
        bench_to_c_str_unchecked(b, s_short)
    }

    #[bench]
    fn bench_to_c_str_unchecked_medium(b: &mut Bencher) {
        bench_to_c_str_unchecked(b, s_medium)
    }

    #[bench]
    fn bench_to_c_str_unchecked_long(b: &mut Bencher) {
        bench_to_c_str_unchecked(b, s_long)
    }

    fn bench_with_c_str(b: &mut Bencher, s: &str) {
        b.iter(|| {
            s.with_c_str(|c_str_buf| check(s, c_str_buf))
        })
    }

    #[bench]
    fn bench_with_c_str_short(b: &mut Bencher) {
        bench_with_c_str(b, s_short)
    }

    #[bench]
    fn bench_with_c_str_medium(b: &mut Bencher) {
        bench_with_c_str(b, s_medium)
    }

    #[bench]
    fn bench_with_c_str_long(b: &mut Bencher) {
        bench_with_c_str(b, s_long)
    }

    fn bench_with_c_str_unchecked(b: &mut Bencher, s: &str) {
        b.iter(|| {
            unsafe {
                s.with_c_str_unchecked(|c_str_buf| check(s, c_str_buf))
            }
        })
    }

    #[bench]
    fn bench_with_c_str_unchecked_short(b: &mut Bencher) {
        bench_with_c_str_unchecked(b, s_short)
    }

    #[bench]
    fn bench_with_c_str_unchecked_medium(b: &mut Bencher) {
        bench_with_c_str_unchecked(b, s_medium)
    }

    #[bench]
    fn bench_with_c_str_unchecked_long(b: &mut Bencher) {
        bench_with_c_str_unchecked(b, s_long)
    }
}
