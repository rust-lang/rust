// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use iter::IteratorExt;
use libc;
use mem;
use ops::Deref;
use slice::{self, SliceExt, AsSlice};
use string::String;
use vec::Vec;

/// A type representing a C-compatible string
///
/// This type serves the primary purpose of being able to generate a
/// C-compatible string from a Rust byte slice or vector. An instance of this
/// type is a static guarantee that the underlying bytes contain no interior 0
/// bytes and the final byte is 0.
///
/// A `CString` is created from either a byte slice or a byte vector. After
/// being created, a `CString` predominately inherits all of its methods from
/// the `Deref` implementation to `[libc::c_char]`. Note that the underlying
/// array is represented as an array of `libc::c_char` as opposed to `u8`. A
/// `u8` slice can be obtained with the `as_bytes` method.  Slices produced from
/// a `CString` do *not* contain the trailing nul terminator unless otherwise
/// specified.
///
/// # Example
///
/// ```no_run
/// # extern crate libc;
/// # fn main() {
/// use std::ffi::CString;
/// use libc;
///
/// extern {
///     fn my_printer(s: *const libc::c_char);
/// }
///
/// let to_print = "Hello, world!";
/// let c_to_print = CString::from_slice(to_print.as_bytes());
/// unsafe {
///     my_printer(c_to_print.as_ptr());
/// }
/// # }
/// ```
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct CString {
    inner: Vec<libc::c_char>,
}

impl CString {
    /// Create a new C-compatible string from a byte slice.
    ///
    /// This method will copy the data of the slice provided into a new
    /// allocation, ensuring that there is a trailing 0 byte.
    ///
    /// # Panics
    ///
    /// This function will panic if there are any 0 bytes already in the slice
    /// provided.
    pub fn from_slice(v: &[u8]) -> CString {
        CString::from_vec(v.to_vec())
    }

    /// Create a C-compatible string from a byte vector.
    ///
    /// This method will consume ownership of the provided vector, appending a 0
    /// byte to the end after verifying that there are no interior 0 bytes.
    ///
    /// # Panics
    ///
    /// This function will panic if there are any 0 bytes already in the vector
    /// provided.
    pub fn from_vec(v: Vec<u8>) -> CString {
        assert!(!v.iter().any(|&x| x == 0));
        unsafe { CString::from_vec_unchecked(v) }
    }

    /// Create a C-compatible string from a byte vector without checking for
    /// interior 0 bytes.
    ///
    /// This method is equivalent to `from_vec` except that no runtime assertion
    /// is made that `v` contains no 0 bytes.
    pub unsafe fn from_vec_unchecked(mut v: Vec<u8>) -> CString {
        v.push(0);
        CString { inner: mem::transmute(v) }
    }

    /// Create a view into this C string which includes the trailing nul
    /// terminator at the end of the string.
    pub fn as_slice_with_nul(&self) -> &[libc::c_char] { self.inner.as_slice() }

    /// Similar to the `as_slice` method, but returns a `u8` slice instead of a
    /// `libc::c_char` slice.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { mem::transmute(self.as_slice()) }
    }

    /// Equivalent to `as_slice_with_nul` except that the type returned is a
    /// `u8` slice instead of a `libc::c_char` slice.
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        unsafe { mem::transmute(self.as_slice_with_nul()) }
    }
}

impl Deref for CString {
    type Target = [libc::c_char];

    fn deref(&self) -> &[libc::c_char] {
        &self.inner[..(self.inner.len() - 1)]
    }
}

#[stable]
impl fmt::Debug for CString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        String::from_utf8_lossy(self.as_bytes()).fmt(f)
    }
}

/// Interpret a C string as a byte slice.
///
/// This function will calculate the length of the C string provided, and it
/// will then return a corresponding slice for the contents of the C string not
/// including the nul terminator.
///
/// This function will tie the lifetime of the returned slice to the lifetime of
/// the pointer provided. This is done to help prevent the slice from escaping
/// the lifetime of the pointer itself. If a longer lifetime is needed, then
/// `mem::copy_lifetime` should be used.
///
/// This function is unsafe because there is no guarantee of the validity of the
/// pointer `raw` or a guarantee that a nul terminator will be found.
///
/// # Example
///
/// ```no_run
/// # extern crate libc;
/// # fn main() {
/// use std::ffi;
/// use std::str;
/// use libc;
///
/// extern {
///     fn my_string() -> *const libc::c_char;
/// }
///
/// unsafe {
///     let to_print = my_string();
///     let slice = ffi::c_str_to_bytes(&to_print);
///     println!("string returned: {}", str::from_utf8(slice).unwrap());
/// }
/// # }
/// ```
pub unsafe fn c_str_to_bytes<'a>(raw: &'a *const libc::c_char) -> &'a [u8] {
    let len = libc::strlen(*raw);
    slice::from_raw_buf(&*(raw as *const _ as *const *const u8), len as uint)
}

/// Interpret a C string as a byte slice with the nul terminator.
///
/// This function is identical to `from_raw_buf` except that the returned slice
/// will include the nul terminator of the string.
pub unsafe fn c_str_to_bytes_with_nul<'a>(raw: &'a *const libc::c_char) -> &'a [u8] {
    let len = libc::strlen(*raw) + 1;
    slice::from_raw_buf(&*(raw as *const _ as *const *const u8), len as uint)
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use super::*;
    use libc;
    use mem;

    #[test]
    fn c_to_rust() {
        let data = b"123\0";
        let ptr = data.as_ptr() as *const libc::c_char;
        unsafe {
            assert_eq!(c_str_to_bytes(&ptr), b"123");
            assert_eq!(c_str_to_bytes_with_nul(&ptr), b"123\0");
        }
    }

    #[test]
    fn simple() {
        let s = CString::from_slice(b"1234");
        assert_eq!(s.as_bytes(), b"1234");
        assert_eq!(s.as_bytes_with_nul(), b"1234\0");
        unsafe {
            assert_eq!(s.as_slice(),
                       mem::transmute::<_, &[libc::c_char]>(b"1234"));
            assert_eq!(s.as_slice_with_nul(),
                       mem::transmute::<_, &[libc::c_char]>(b"1234\0"));
        }
    }

    #[should_fail] #[test]
    fn build_with_zero1() { CString::from_slice(b"\0"); }
    #[should_fail] #[test]
    fn build_with_zero2() { CString::from_vec(vec![0]); }

    #[test]
    fn build_with_zero3() {
        unsafe {
            let s = CString::from_vec_unchecked(vec![0]);
            assert_eq!(s.as_bytes(), b"\0");
        }
    }

    #[test]
    fn formatted() {
        let s = CString::from_slice(b"12");
        assert_eq!(format!("{:?}", s), "\"12\"");
    }
}
