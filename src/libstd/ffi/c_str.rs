// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use error::{Error, FromError};
use fmt;
use io;
use iter::IteratorExt;
use libc;
use mem;
use old_io;
use ops::Deref;
use option::Option::{self, Some, None};
use result::Result::{self, Ok, Err};
use slice::{self, SliceExt};
use str::StrExt;
use string::String;
use vec::Vec;

/// A type representing an owned C-compatible string
///
/// This type serves the primary purpose of being able to safely generate a
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
/// let to_print = b"Hello, world!";
/// let c_to_print = CString::new(to_print).unwrap();
/// unsafe {
///     my_printer(c_to_print.as_ptr());
/// }
/// # }
/// ```
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct CString {
    inner: Vec<u8>,
}

/// Representation of a borrowed C string.
///
/// This dynamically sized type is only safely constructed via a borrowed
/// version of an instance of `CString`. This type can be constructed from a raw
/// C string as well and represents a C string borrowed from another location.
///
/// Note that this structure is **not** `repr(C)` and is not recommended to be
/// placed in the signatures of FFI functions. Instead safe wrappers of FFI
/// functions may leverage the unsafe `from_ptr` constructor to provide a safe
/// interface to other consumers.
///
/// # Examples
///
/// Inspecting a foreign C string
///
/// ```no_run
/// extern crate libc;
/// use std::ffi::CStr;
///
/// extern { fn my_string() -> *const libc::c_char; }
///
/// fn main() {
///     unsafe {
///         let slice = CStr::from_ptr(my_string());
///         println!("string length: {}", slice.to_bytes().len());
///     }
/// }
/// ```
///
/// Passing a Rust-originating C string
///
/// ```no_run
/// extern crate libc;
/// use std::ffi::{CString, CStr};
///
/// fn work(data: &CStr) {
///     extern { fn work_with(data: *const libc::c_char); }
///
///     unsafe { work_with(data.as_ptr()) }
/// }
///
/// fn main() {
///     let s = CString::new("data data data data").unwrap();
///     work(&s);
/// }
/// ```
#[derive(Hash)]
pub struct CStr {
    inner: [libc::c_char]
}

/// An error returned from `CString::new` to indicate that a nul byte was found
/// in the vector provided.
#[derive(Clone, PartialEq, Debug)]
pub struct NulError(usize, Vec<u8>);

/// A conversion trait used by the constructor of `CString` for types that can
/// be converted to a vector of bytes.
pub trait IntoBytes {
    /// Consumes this container, returning a vector of bytes.
    fn into_bytes(self) -> Vec<u8>;
}

impl CString {
    /// Create a new C-compatible string from a container of bytes.
    ///
    /// This method will consume the provided data and use the underlying bytes
    /// to construct a new string, ensuring that there is a trailing 0 byte.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate libc;
    /// use std::ffi::CString;
    ///
    /// extern { fn puts(s: *const libc::c_char); }
    ///
    /// fn main() {
    ///     let to_print = CString::new("Hello!").unwrap();
    ///     unsafe {
    ///         puts(to_print.as_ptr());
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if the bytes yielded contain an
    /// internal 0 byte. The error returned will contain the bytes as well as
    /// the position of the nul byte.
    pub fn new<T: IntoBytes>(t: T) -> Result<CString, NulError> {
        let bytes = t.into_bytes();
        match bytes.iter().position(|x| *x == 0) {
            Some(i) => Err(NulError(i, bytes)),
            None => Ok(unsafe { CString::from_vec_unchecked(bytes) }),
        }
    }

    /// Create a new C-compatible string from a byte slice.
    ///
    /// This method will copy the data of the slice provided into a new
    /// allocation, ensuring that there is a trailing 0 byte.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate libc;
    /// use std::ffi::CString;
    ///
    /// extern { fn puts(s: *const libc::c_char); }
    ///
    /// fn main() {
    ///     let to_print = CString::new("Hello!").unwrap();
    ///     unsafe {
    ///         puts(to_print.as_ptr());
    ///     }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the provided slice contains any
    /// interior nul bytes.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0", reason = "use CString::new instead")]
    #[allow(deprecated)]
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
    /// This function will panic if the provided slice contains any
    /// interior nul bytes.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0", reason = "use CString::new instead")]
    pub fn from_vec(v: Vec<u8>) -> CString {
        match v.iter().position(|x| *x == 0) {
            Some(i) => panic!("null byte found in slice at: {}", i),
            None => unsafe { CString::from_vec_unchecked(v) },
        }
    }

    /// Create a C-compatible string from a byte vector without checking for
    /// interior 0 bytes.
    ///
    /// This method is equivalent to `from_vec` except that no runtime assertion
    /// is made that `v` contains no 0 bytes.
    pub unsafe fn from_vec_unchecked(mut v: Vec<u8>) -> CString {
        v.push(0);
        CString { inner: v }
    }

    /// Returns the contents of this `CString` as a slice of bytes.
    ///
    /// The returned slice does **not** contain the trailing nul separator and
    /// it is guaranteet to not have any interior nul bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner[..self.inner.len() - 1]
    }

    /// Equivalent to the `as_bytes` function except that the returned slice
    /// includes the trailing nul byte.
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        &self.inner
    }
}

impl Deref for CString {
    type Target = CStr;

    fn deref(&self) -> &CStr {
        unsafe { mem::transmute(self.as_bytes_with_nul()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for CString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&String::from_utf8_lossy(self.as_bytes()), f)
    }
}

impl NulError {
    /// Returns the position of the nul byte in the slice that was provided to
    /// `CString::from_vec`.
    pub fn nul_position(&self) -> usize { self.0 }

    /// Consumes this error, returning the underlying vector of bytes which
    /// generated the error in the first place.
    pub fn into_vec(self) -> Vec<u8> { self.1 }
}

impl Error for NulError {
    fn description(&self) -> &str { "nul byte found in data" }
}

impl fmt::Display for NulError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "nul byte found in provided data at position: {}", self.0)
    }
}

impl FromError<NulError> for io::Error {
    fn from_error(_: NulError) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput,
                       "data provided contains a nul byte", None)
    }
}

impl FromError<NulError> for old_io::IoError {
    fn from_error(_: NulError) -> old_io::IoError {
        old_io::IoError {
            kind: old_io::IoErrorKind::InvalidInput,
            desc: "data provided contains a nul byte",
            detail: None
        }
    }
}

impl CStr {
    /// Cast a raw C string to a safe C string wrapper.
    ///
    /// This function will cast the provided `ptr` to the `CStr` wrapper which
    /// allows inspection and interoperation of non-owned C strings. This method
    /// is unsafe for a number of reasons:
    ///
    /// * There is no guarantee to the validity of `ptr`
    /// * The returned lifetime is not guaranteed to be the actual lifetime of
    ///   `ptr`
    /// * There is no guarantee that the memory pointed to by `ptr` contains a
    ///   valid nul terminator byte at the end of the string.
    ///
    /// > **Note**: This operation is intended to be a 0-cost cast but it is
    /// > currently implemented with an up-front calculation of the length of
    /// > the string. This is not guaranteed to always be the case.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # extern crate libc;
    /// # fn main() {
    /// use std::ffi::CStr;
    /// use std::str;
    /// use libc;
    ///
    /// extern {
    ///     fn my_string() -> *const libc::c_char;
    /// }
    ///
    /// unsafe {
    ///     let slice = CStr::from_ptr(my_string());
    ///     println!("string returned: {}",
    ///              str::from_utf8(slice.to_bytes()).unwrap());
    /// }
    /// # }
    /// ```
    pub unsafe fn from_ptr<'a>(ptr: *const libc::c_char) -> &'a CStr {
        let len = libc::strlen(ptr);
        mem::transmute(slice::from_raw_parts(ptr, len as usize + 1))
    }

    /// Return the inner pointer to this C string.
    ///
    /// The returned pointer will be valid for as long as `self` is and points
    /// to a continguous region of memory terminated with a 0 byte to represent
    /// the end of the string.
    pub fn as_ptr(&self) -> *const libc::c_char {
        self.inner.as_ptr()
    }

    /// Convert this C string to a byte slice.
    ///
    /// This function will calculate the length of this string (which normally
    /// requires a linear amount of work to be done) and then return the
    /// resulting slice of `u8` elements.
    ///
    /// The returned slice will **not** contain the trailing nul that this C
    /// string has.
    ///
    /// > **Note**: This method is currently implemented as a 0-cost cast, but
    /// > it is planned to alter its definition in the future to perform the
    /// > length calculation whenever this method is called.
    pub fn to_bytes(&self) -> &[u8] {
        let bytes = self.to_bytes_with_nul();
        &bytes[..bytes.len() - 1]
    }

    /// Convert this C string to a byte slice containing the trailing 0 byte.
    ///
    /// This function is the equivalent of `to_bytes` except that it will retain
    /// the trailing nul instead of chopping it off.
    ///
    /// > **Note**: This method is currently implemented as a 0-cost cast, but
    /// > it is planned to alter its definition in the future to perform the
    /// > length calculation whenever this method is called.
    pub fn to_bytes_with_nul(&self) -> &[u8] {
        unsafe { mem::transmute::<&[libc::c_char], &[u8]>(&self.inner) }
    }
}

impl PartialEq for CStr {
    fn eq(&self, other: &CStr) -> bool {
        self.to_bytes().eq(&other.to_bytes())
    }
}
impl Eq for CStr {}
impl PartialOrd for CStr {
    fn partial_cmp(&self, other: &CStr) -> Option<Ordering> {
        self.to_bytes().partial_cmp(&other.to_bytes())
    }
}
impl Ord for CStr {
    fn cmp(&self, other: &CStr) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

/// Deprecated in favor of `CStr`
#[unstable(feature = "std_misc")]
#[deprecated(since = "1.0.0", reason = "use CStr::from_ptr(p).to_bytes() instead")]
pub unsafe fn c_str_to_bytes<'a>(raw: &'a *const libc::c_char) -> &'a [u8] {
    let len = libc::strlen(*raw);
    slice::from_raw_parts(*(raw as *const _ as *const *const u8), len as usize)
}

/// Deprecated in favor of `CStr`
#[unstable(feature = "std_misc")]
#[deprecated(since = "1.0.0",
             reason = "use CStr::from_ptr(p).to_bytes_with_nul() instead")]
pub unsafe fn c_str_to_bytes_with_nul<'a>(raw: &'a *const libc::c_char)
                                          -> &'a [u8] {
    let len = libc::strlen(*raw) + 1;
    slice::from_raw_parts(*(raw as *const _ as *const *const u8), len as usize)
}

impl<'a> IntoBytes for &'a str {
    fn into_bytes(self) -> Vec<u8> { self.as_bytes().to_vec() }
}
impl<'a> IntoBytes for &'a [u8] {
    fn into_bytes(self) -> Vec<u8> { self.to_vec() }
}
impl IntoBytes for String {
    fn into_bytes(self) -> Vec<u8> { self.into_bytes() }
}
impl IntoBytes for Vec<u8> {
    fn into_bytes(self) -> Vec<u8> { self }
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
        let s = CString::new(b"1234").unwrap();
        assert_eq!(s.as_bytes(), b"1234");
        assert_eq!(s.as_bytes_with_nul(), b"1234\0");
    }

    #[test]
    fn build_with_zero1() {
        assert!(CString::new(b"\0").is_err());
    }
    #[test]
    fn build_with_zero2() {
        assert!(CString::new(vec![0]).is_err());
    }

    #[test]
    fn build_with_zero3() {
        unsafe {
            let s = CString::from_vec_unchecked(vec![0]);
            assert_eq!(s.as_bytes(), b"\0");
        }
    }

    #[test]
    fn formatted() {
        let s = CString::new(b"12").unwrap();
        assert_eq!(format!("{:?}", s), "\"12\"");
    }

    #[test]
    fn borrowed() {
        unsafe {
            let s = CStr::from_ptr(b"12\0".as_ptr() as *const _);
            assert_eq!(s.to_bytes(), b"12");
            assert_eq!(s.to_bytes_with_nul(), b"12\0");
        }
    }
}
