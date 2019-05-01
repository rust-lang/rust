use crate::ascii;
use crate::borrow::{Cow, Borrow};
use crate::cmp::Ordering;
use crate::error::Error;
use crate::fmt::{self, Write};
use crate::io;
use crate::mem;
use crate::memchr;
use crate::ops;
use crate::os::raw::c_char;
use crate::ptr;
use crate::rc::Rc;
use crate::slice;
use crate::str::{self, Utf8Error};
use crate::sync::Arc;
use crate::sys;

/// A type representing an owned, C-compatible, nul-terminated string with no nul bytes in the
/// middle.
///
/// This type serves the purpose of being able to safely generate a
/// C-compatible string from a Rust byte slice or vector. An instance of this
/// type is a static guarantee that the underlying bytes contain no interior 0
/// bytes ("nul characters") and that the final byte is 0 ("nul terminator").
///
/// `CString` is to [`&CStr`] as [`String`] is to [`&str`]: the former
/// in each pair are owned strings; the latter are borrowed
/// references.
///
/// # Creating a `CString`
///
/// A `CString` is created from either a byte slice or a byte vector,
/// or anything that implements [`Into`]`<`[`Vec`]`<`[`u8`]`>>` (for
/// example, you can build a `CString` straight out of a [`String`] or
/// a [`&str`], since both implement that trait).
///
/// The [`new`] method will actually check that the provided `&[u8]`
/// does not have 0 bytes in the middle, and return an error if it
/// finds one.
///
/// # Extracting a raw pointer to the whole C string
///
/// `CString` implements a [`as_ptr`] method through the [`Deref`]
/// trait. This method will give you a `*const c_char` which you can
/// feed directly to extern functions that expect a nul-terminated
/// string, like C's `strdup()`. Notice that [`as_ptr`] returns a
/// read-only pointer; if the C code writes to it, that causes
/// undefined behavior.
///
/// # Extracting a slice of the whole C string
///
/// Alternatively, you can obtain a `&[`[`u8`]`]` slice from a
/// `CString` with the [`as_bytes`] method. Slices produced in this
/// way do *not* contain the trailing nul terminator. This is useful
/// when you will be calling an extern function that takes a `*const
/// u8` argument which is not necessarily nul-terminated, plus another
/// argument with the length of the string — like C's `strndup()`.
/// You can of course get the slice's length with its
/// [`len`][slice.len] method.
///
/// If you need a `&[`[`u8`]`]` slice *with* the nul terminator, you
/// can use [`as_bytes_with_nul`] instead.
///
/// Once you have the kind of slice you need (with or without a nul
/// terminator), you can call the slice's own
/// [`as_ptr`][slice.as_ptr] method to get a read-only raw pointer to pass to
/// extern functions. See the documentation for that function for a
/// discussion on ensuring the lifetime of the raw pointer.
///
/// [`Into`]: ../convert/trait.Into.html
/// [`Vec`]: ../vec/struct.Vec.html
/// [`String`]: ../string/struct.String.html
/// [`&str`]: ../primitive.str.html
/// [`u8`]: ../primitive.u8.html
/// [`new`]: #method.new
/// [`as_bytes`]: #method.as_bytes
/// [`as_bytes_with_nul`]: #method.as_bytes_with_nul
/// [`as_ptr`]: #method.as_ptr
/// [slice.as_ptr]: ../primitive.slice.html#method.as_ptr
/// [slice.len]: ../primitive.slice.html#method.len
/// [`Deref`]: ../ops/trait.Deref.html
/// [`CStr`]: struct.CStr.html
/// [`&CStr`]: struct.CStr.html
///
/// # Examples
///
/// ```ignore (extern-declaration)
/// # fn main() {
/// use std::ffi::CString;
/// use std::os::raw::c_char;
///
/// extern {
///     fn my_printer(s: *const c_char);
/// }
///
/// // We are certain that our string doesn't have 0 bytes in the middle,
/// // so we can .expect()
/// let c_to_print = CString::new("Hello, world!").expect("CString::new failed");
/// unsafe {
///     my_printer(c_to_print.as_ptr());
/// }
/// # }
/// ```
///
/// # Safety
///
/// `CString` is intended for working with traditional C-style strings
/// (a sequence of non-nul bytes terminated by a single nul byte); the
/// primary use case for these kinds of strings is interoperating with C-like
/// code. Often you will need to transfer ownership to/from that external
/// code. It is strongly recommended that you thoroughly read through the
/// documentation of `CString` before use, as improper ownership management
/// of `CString` instances can lead to invalid memory accesses, memory leaks,
/// and other memory errors.

#[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct CString {
    // Invariant 1: the slice ends with a zero byte and has a length of at least one.
    // Invariant 2: the slice contains only one zero byte.
    // Improper usage of unsafe function can break Invariant 2, but not Invariant 1.
    inner: Box<[u8]>,
}

/// Representation of a borrowed C string.
///
/// This type represents a borrowed reference to a nul-terminated
/// array of bytes. It can be constructed safely from a `&[`[`u8`]`]`
/// slice, or unsafely from a raw `*const c_char`. It can then be
/// converted to a Rust [`&str`] by performing UTF-8 validation, or
/// into an owned [`CString`].
///
/// `&CStr` is to [`CString`] as [`&str`] is to [`String`]: the former
/// in each pair are borrowed references; the latter are owned
/// strings.
///
/// Note that this structure is **not** `repr(C)` and is not recommended to be
/// placed in the signatures of FFI functions. Instead, safe wrappers of FFI
/// functions may leverage the unsafe [`from_ptr`] constructor to provide a safe
/// interface to other consumers.
///
/// # Examples
///
/// Inspecting a foreign C string:
///
/// ```ignore (extern-declaration)
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// extern { fn my_string() -> *const c_char; }
///
/// unsafe {
///     let slice = CStr::from_ptr(my_string());
///     println!("string buffer size without nul terminator: {}", slice.to_bytes().len());
/// }
/// ```
///
/// Passing a Rust-originating C string:
///
/// ```ignore (extern-declaration)
/// use std::ffi::{CString, CStr};
/// use std::os::raw::c_char;
///
/// fn work(data: &CStr) {
///     extern { fn work_with(data: *const c_char); }
///
///     unsafe { work_with(data.as_ptr()) }
/// }
///
/// let s = CString::new("data data data data").expect("CString::new failed");
/// work(&s);
/// ```
///
/// Converting a foreign C string into a Rust [`String`]:
///
/// ```ignore (extern-declaration)
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// extern { fn my_string() -> *const c_char; }
///
/// fn my_string_safe() -> String {
///     unsafe {
///         CStr::from_ptr(my_string()).to_string_lossy().into_owned()
///     }
/// }
///
/// println!("string: {}", my_string_safe());
/// ```
///
/// [`u8`]: ../primitive.u8.html
/// [`&str`]: ../primitive.str.html
/// [`String`]: ../string/struct.String.html
/// [`CString`]: struct.CString.html
/// [`from_ptr`]: #method.from_ptr
#[derive(Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct CStr {
    // FIXME: this should not be represented with a DST slice but rather with
    //        just a raw `c_char` along with some form of marker to make
    //        this an unsized type. Essentially `sizeof(&CStr)` should be the
    //        same as `sizeof(&c_char)` but `CStr` should be an unsized type.
    inner: [c_char]
}

/// An error indicating that an interior nul byte was found.
///
/// While Rust strings may contain nul bytes in the middle, C strings
/// can't, as that byte would effectively truncate the string.
///
/// This error is created by the [`new`][`CString::new`] method on
/// [`CString`]. See its documentation for more.
///
/// [`CString`]: struct.CString.html
/// [`CString::new`]: struct.CString.html#method.new
///
/// # Examples
///
/// ```
/// use std::ffi::{CString, NulError};
///
/// let _: NulError = CString::new(b"f\0oo".to_vec()).unwrap_err();
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct NulError(usize, Vec<u8>);

/// An error indicating that a nul byte was not in the expected position.
///
/// The slice used to create a [`CStr`] must have one and only one nul
/// byte at the end of the slice.
///
/// This error is created by the
/// [`from_bytes_with_nul`][`CStr::from_bytes_with_nul`] method on
/// [`CStr`]. See its documentation for more.
///
/// [`CStr`]: struct.CStr.html
/// [`CStr::from_bytes_with_nul`]: struct.CStr.html#method.from_bytes_with_nul
///
/// # Examples
///
/// ```
/// use std::ffi::{CStr, FromBytesWithNulError};
///
/// let _: FromBytesWithNulError = CStr::from_bytes_with_nul(b"f\0oo").unwrap_err();
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "cstr_from_bytes", since = "1.10.0")]
pub struct FromBytesWithNulError {
    kind: FromBytesWithNulErrorKind,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum FromBytesWithNulErrorKind {
    InteriorNul(usize),
    NotNulTerminated,
}

impl FromBytesWithNulError {
    fn interior_nul(pos: usize) -> FromBytesWithNulError {
        FromBytesWithNulError {
            kind: FromBytesWithNulErrorKind::InteriorNul(pos),
        }
    }
    fn not_nul_terminated() -> FromBytesWithNulError {
        FromBytesWithNulError {
            kind: FromBytesWithNulErrorKind::NotNulTerminated,
        }
    }
}

/// An error indicating invalid UTF-8 when converting a [`CString`] into a [`String`].
///
/// `CString` is just a wrapper over a buffer of bytes with a nul
/// terminator; [`into_string`][`CString::into_string`] performs UTF-8
/// validation on those bytes and may return this error.
///
/// This `struct` is created by the
/// [`into_string`][`CString::into_string`] method on [`CString`]. See
/// its documentation for more.
///
/// [`String`]: ../string/struct.String.html
/// [`CString`]: struct.CString.html
/// [`CString::into_string`]: struct.CString.html#method.into_string
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "cstring_into", since = "1.7.0")]
pub struct IntoStringError {
    inner: CString,
    error: Utf8Error,
}

impl CString {
    /// Creates a new C-compatible string from a container of bytes.
    ///
    /// This function will consume the provided data and use the
    /// underlying bytes to construct a new string, ensuring that
    /// there is a trailing 0 byte. This trailing 0 byte will be
    /// appended by this function; the provided data should *not*
    /// contain any 0 bytes in it.
    ///
    /// # Examples
    ///
    /// ```ignore (extern-declaration)
    /// use std::ffi::CString;
    /// use std::os::raw::c_char;
    ///
    /// extern { fn puts(s: *const c_char); }
    ///
    /// let to_print = CString::new("Hello!").expect("CString::new failed");
    /// unsafe {
    ///     puts(to_print.as_ptr());
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if the supplied bytes contain an
    /// internal 0 byte. The [`NulError`] returned will contain the bytes as well as
    /// the position of the nul byte.
    ///
    /// [`NulError`]: struct.NulError.html
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<T: Into<Vec<u8>>>(t: T) -> Result<CString, NulError> {
        Self::_new(t.into())
    }

    fn _new(bytes: Vec<u8>) -> Result<CString, NulError> {
        match memchr::memchr(0, &bytes) {
            Some(i) => Err(NulError(i, bytes)),
            None => Ok(unsafe { CString::from_vec_unchecked(bytes) }),
        }
    }

    /// Creates a C-compatible string by consuming a byte vector,
    /// without checking for interior 0 bytes.
    ///
    /// This method is equivalent to [`new`] except that no runtime assertion
    /// is made that `v` contains no 0 bytes, and it requires an actual
    /// byte vector, not anything that can be converted to one with Into.
    ///
    /// [`new`]: #method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let raw = b"foo".to_vec();
    /// unsafe {
    ///     let c_string = CString::from_vec_unchecked(raw);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_vec_unchecked(mut v: Vec<u8>) -> CString {
        v.reserve_exact(1);
        v.push(0);
        CString { inner: v.into_boxed_slice() }
    }

    /// Retakes ownership of a `CString` that was transferred to C via [`into_raw`].
    ///
    /// Additionally, the length of the string will be recalculated from the pointer.
    ///
    /// # Safety
    ///
    /// This should only ever be called with a pointer that was earlier
    /// obtained by calling [`into_raw`] on a `CString`. Other usage (e.g., trying to take
    /// ownership of a string that was allocated by foreign code) is likely to lead
    /// to undefined behavior or allocator corruption.
    ///
    /// > **Note:** If you need to borrow a string that was allocated by
    /// > foreign code, use [`CStr`]. If you need to take ownership of
    /// > a string that was allocated by foreign code, you will need to
    /// > make your own provisions for freeing it appropriately, likely
    /// > with the foreign code's API to do that.
    ///
    /// [`into_raw`]: #method.into_raw
    /// [`CStr`]: struct.CStr.html
    ///
    /// # Examples
    ///
    /// Creates a `CString`, pass ownership to an `extern` function (via raw pointer), then retake
    /// ownership with `from_raw`:
    ///
    /// ```ignore (extern-declaration)
    /// use std::ffi::CString;
    /// use std::os::raw::c_char;
    ///
    /// extern {
    ///     fn some_extern_function(s: *mut c_char);
    /// }
    ///
    /// let c_string = CString::new("Hello!").expect("CString::new failed");
    /// let raw = c_string.into_raw();
    /// unsafe {
    ///     some_extern_function(raw);
    ///     let c_string = CString::from_raw(raw);
    /// }
    /// ```
    #[stable(feature = "cstr_memory", since = "1.4.0")]
    pub unsafe fn from_raw(ptr: *mut c_char) -> CString {
        let len = sys::strlen(ptr) + 1; // Including the NUL byte
        let slice = slice::from_raw_parts_mut(ptr, len as usize);
        CString { inner: Box::from_raw(slice as *mut [c_char] as *mut [u8]) }
    }

    /// Consumes the `CString` and transfers ownership of the string to a C caller.
    ///
    /// The pointer which this function returns must be returned to Rust and reconstituted using
    /// [`from_raw`] to be properly deallocated. Specifically, one
    /// should *not* use the standard C `free()` function to deallocate
    /// this string.
    ///
    /// Failure to call [`from_raw`] will lead to a memory leak.
    ///
    /// [`from_raw`]: #method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::new("foo").expect("CString::new failed");
    ///
    /// let ptr = c_string.into_raw();
    ///
    /// unsafe {
    ///     assert_eq!(b'f', *ptr as u8);
    ///     assert_eq!(b'o', *ptr.offset(1) as u8);
    ///     assert_eq!(b'o', *ptr.offset(2) as u8);
    ///     assert_eq!(b'\0', *ptr.offset(3) as u8);
    ///
    ///     // retake pointer to free memory
    ///     let _ = CString::from_raw(ptr);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "cstr_memory", since = "1.4.0")]
    pub fn into_raw(self) -> *mut c_char {
        Box::into_raw(self.into_inner()) as *mut c_char
    }

    /// Converts the `CString` into a [`String`] if it contains valid UTF-8 data.
    ///
    /// On failure, ownership of the original `CString` is returned.
    ///
    /// [`String`]: ../string/struct.String.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let valid_utf8 = vec![b'f', b'o', b'o'];
    /// let cstring = CString::new(valid_utf8).expect("CString::new failed");
    /// assert_eq!(cstring.into_string().expect("into_string() call failed"), "foo");
    ///
    /// let invalid_utf8 = vec![b'f', 0xff, b'o', b'o'];
    /// let cstring = CString::new(invalid_utf8).expect("CString::new failed");
    /// let err = cstring.into_string().err().expect("into_string().err() failed");
    /// assert_eq!(err.utf8_error().valid_up_to(), 1);
    /// ```

    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_string(self) -> Result<String, IntoStringError> {
        String::from_utf8(self.into_bytes())
            .map_err(|e| IntoStringError {
                error: e.utf8_error(),
                inner: unsafe { CString::from_vec_unchecked(e.into_bytes()) },
            })
    }

    /// Consumes the `CString` and returns the underlying byte buffer.
    ///
    /// The returned buffer does **not** contain the trailing nul
    /// terminator, and it is guaranteed to not have any interior nul
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::new("foo").expect("CString::new failed");
    /// let bytes = c_string.into_bytes();
    /// assert_eq!(bytes, vec![b'f', b'o', b'o']);
    /// ```
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_bytes(self) -> Vec<u8> {
        let mut vec = self.into_inner().into_vec();
        let _nul = vec.pop();
        debug_assert_eq!(_nul, Some(0u8));
        vec
    }

    /// Equivalent to the [`into_bytes`] function except that the returned vector
    /// includes the trailing nul terminator.
    ///
    /// [`into_bytes`]: #method.into_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::new("foo").expect("CString::new failed");
    /// let bytes = c_string.into_bytes_with_nul();
    /// assert_eq!(bytes, vec![b'f', b'o', b'o', b'\0']);
    /// ```
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_bytes_with_nul(self) -> Vec<u8> {
        self.into_inner().into_vec()
    }

    /// Returns the contents of this `CString` as a slice of bytes.
    ///
    /// The returned slice does **not** contain the trailing nul
    /// terminator, and it is guaranteed to not have any interior nul
    /// bytes. If you need the nul terminator, use
    /// [`as_bytes_with_nul`] instead.
    ///
    /// [`as_bytes_with_nul`]: #method.as_bytes_with_nul
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::new("foo").expect("CString::new failed");
    /// let bytes = c_string.as_bytes();
    /// assert_eq!(bytes, &[b'f', b'o', b'o']);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner[..self.inner.len() - 1]
    }

    /// Equivalent to the [`as_bytes`] function except that the returned slice
    /// includes the trailing nul terminator.
    ///
    /// [`as_bytes`]: #method.as_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::new("foo").expect("CString::new failed");
    /// let bytes = c_string.as_bytes_with_nul();
    /// assert_eq!(bytes, &[b'f', b'o', b'o', b'\0']);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        &self.inner
    }

    /// Extracts a [`CStr`] slice containing the entire string.
    ///
    /// [`CStr`]: struct.CStr.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{CString, CStr};
    ///
    /// let c_string = CString::new(b"foo".to_vec()).expect("CString::new failed");
    /// let c_str = c_string.as_c_str();
    /// assert_eq!(c_str,
    ///            CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed"));
    /// ```
    #[inline]
    #[stable(feature = "as_c_str", since = "1.20.0")]
    pub fn as_c_str(&self) -> &CStr {
        &*self
    }

    /// Converts this `CString` into a boxed [`CStr`].
    ///
    /// [`CStr`]: struct.CStr.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{CString, CStr};
    ///
    /// let c_string = CString::new(b"foo".to_vec()).expect("CString::new failed");
    /// let boxed = c_string.into_boxed_c_str();
    /// assert_eq!(&*boxed,
    ///            CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed"));
    /// ```
    #[stable(feature = "into_boxed_c_str", since = "1.20.0")]
    pub fn into_boxed_c_str(self) -> Box<CStr> {
        unsafe { Box::from_raw(Box::into_raw(self.into_inner()) as *mut CStr) }
    }

    /// Bypass "move out of struct which implements [`Drop`] trait" restriction.
    ///
    /// [`Drop`]: ../ops/trait.Drop.html
    fn into_inner(self) -> Box<[u8]> {
        unsafe {
            let result = ptr::read(&self.inner);
            mem::forget(self);
            result
        }
    }
}

// Turns this `CString` into an empty string to prevent
// memory unsafe code from working by accident. Inline
// to prevent LLVM from optimizing it away in debug builds.
#[stable(feature = "cstring_drop", since = "1.13.0")]
impl Drop for CString {
    #[inline]
    fn drop(&mut self) {
        unsafe { *self.inner.get_unchecked_mut(0) = 0; }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for CString {
    type Target = CStr;

    #[inline]
    fn deref(&self) -> &CStr {
        unsafe { CStr::from_bytes_with_nul_unchecked(self.as_bytes_with_nul()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for CString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl From<CString> for Vec<u8> {
    /// Converts a [`CString`] into a [`Vec`]`<u8>`.
    ///
    /// The conversion consumes the [`CString`], and removes the terminating NUL byte.
    ///
    /// [`Vec`]: ../vec/struct.Vec.html
    /// [`CString`]: ../ffi/struct.CString.html
    #[inline]
    fn from(s: CString) -> Vec<u8> {
        s.into_bytes()
    }
}

#[stable(feature = "cstr_debug", since = "1.3.0")]
impl fmt::Debug for CStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"")?;
        for byte in self.to_bytes().iter().flat_map(|&b| ascii::escape_default(b)) {
            f.write_char(byte as char)?;
        }
        write!(f, "\"")
    }
}

#[stable(feature = "cstr_default", since = "1.10.0")]
impl Default for &CStr {
    fn default() -> Self {
        const SLICE: &[c_char] = &[0];
        unsafe { CStr::from_ptr(SLICE.as_ptr()) }
    }
}

#[stable(feature = "cstr_default", since = "1.10.0")]
impl Default for CString {
    /// Creates an empty `CString`.
    fn default() -> CString {
        let a: &CStr = Default::default();
        a.to_owned()
    }
}

#[stable(feature = "cstr_borrow", since = "1.3.0")]
impl Borrow<CStr> for CString {
    #[inline]
    fn borrow(&self) -> &CStr { self }
}

#[stable(feature = "cstring_from_cow_cstr", since = "1.28.0")]
impl<'a> From<Cow<'a, CStr>> for CString {
    #[inline]
    fn from(s: Cow<'a, CStr>) -> Self {
        s.into_owned()
    }
}

#[stable(feature = "box_from_c_str", since = "1.17.0")]
impl From<&CStr> for Box<CStr> {
    fn from(s: &CStr) -> Box<CStr> {
        let boxed: Box<[u8]> = Box::from(s.to_bytes_with_nul());
        unsafe { Box::from_raw(Box::into_raw(boxed) as *mut CStr) }
    }
}

#[stable(feature = "c_string_from_box", since = "1.18.0")]
impl From<Box<CStr>> for CString {
    /// Converts a [`Box`]`<CStr>` into a [`CString`] without copying or allocating.
    ///
    /// [`Box`]: ../boxed/struct.Box.html
    /// [`CString`]: ../ffi/struct.CString.html
    #[inline]
    fn from(s: Box<CStr>) -> CString {
        s.into_c_string()
    }
}

#[stable(feature = "more_box_slice_clone", since = "1.29.0")]
impl Clone for Box<CStr> {
    #[inline]
    fn clone(&self) -> Self {
        (**self).into()
    }
}

#[stable(feature = "box_from_c_string", since = "1.20.0")]
impl From<CString> for Box<CStr> {
    /// Converts a [`CString`] into a [`Box`]`<CStr>` without copying or allocating.
    ///
    /// [`CString`]: ../ffi/struct.CString.html
    /// [`Box`]: ../boxed/struct.Box.html
    #[inline]
    fn from(s: CString) -> Box<CStr> {
        s.into_boxed_c_str()
    }
}

#[stable(feature = "cow_from_cstr", since = "1.28.0")]
impl<'a> From<CString> for Cow<'a, CStr> {
    #[inline]
    fn from(s: CString) -> Cow<'a, CStr> {
        Cow::Owned(s)
    }
}

#[stable(feature = "cow_from_cstr", since = "1.28.0")]
impl<'a> From<&'a CStr> for Cow<'a, CStr> {
    #[inline]
    fn from(s: &'a CStr) -> Cow<'a, CStr> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_cstr", since = "1.28.0")]
impl<'a> From<&'a CString> for Cow<'a, CStr> {
    #[inline]
    fn from(s: &'a CString) -> Cow<'a, CStr> {
        Cow::Borrowed(s.as_c_str())
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<CString> for Arc<CStr> {
    /// Converts a [`CString`] into a [`Arc`]`<CStr>` without copying or allocating.
    ///
    /// [`CString`]: ../ffi/struct.CString.html
    /// [`Arc`]: ../sync/struct.Arc.html
    #[inline]
    fn from(s: CString) -> Arc<CStr> {
        let arc: Arc<[u8]> = Arc::from(s.into_inner());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const CStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&CStr> for Arc<CStr> {
    #[inline]
    fn from(s: &CStr) -> Arc<CStr> {
        let arc: Arc<[u8]> = Arc::from(s.to_bytes_with_nul());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const CStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<CString> for Rc<CStr> {
    /// Converts a [`CString`] into a [`Rc`]`<CStr>` without copying or allocating.
    ///
    /// [`CString`]: ../ffi/struct.CString.html
    /// [`Rc`]: ../rc/struct.Rc.html
    #[inline]
    fn from(s: CString) -> Rc<CStr> {
        let rc: Rc<[u8]> = Rc::from(s.into_inner());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const CStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&CStr> for Rc<CStr> {
    #[inline]
    fn from(s: &CStr) -> Rc<CStr> {
        let rc: Rc<[u8]> = Rc::from(s.to_bytes_with_nul());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const CStr) }
    }
}

#[stable(feature = "default_box_extra", since = "1.17.0")]
impl Default for Box<CStr> {
    fn default() -> Box<CStr> {
        let boxed: Box<[u8]> = Box::from([0]);
        unsafe { Box::from_raw(Box::into_raw(boxed) as *mut CStr) }
    }
}

impl NulError {
    /// Returns the position of the nul byte in the slice that caused
    /// [`CString::new`] to fail.
    ///
    /// [`CString::new`]: struct.CString.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let nul_error = CString::new("foo\0bar").unwrap_err();
    /// assert_eq!(nul_error.nul_position(), 3);
    ///
    /// let nul_error = CString::new("foo bar\0").unwrap_err();
    /// assert_eq!(nul_error.nul_position(), 7);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn nul_position(&self) -> usize { self.0 }

    /// Consumes this error, returning the underlying vector of bytes which
    /// generated the error in the first place.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let nul_error = CString::new("foo\0bar").unwrap_err();
    /// assert_eq!(nul_error.into_vec(), b"foo\0bar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_vec(self) -> Vec<u8> { self.1 }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for NulError {
    fn description(&self) -> &str { "nul byte found in data" }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for NulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nul byte found in provided data at position: {}", self.0)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<NulError> for io::Error {
    /// Converts a [`NulError`] into a [`io::Error`].
    ///
    /// [`NulError`]: ../ffi/struct.NulError.html
    /// [`io::Error`]: ../io/struct.Error.html
    fn from(_: NulError) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput,
                       "data provided contains a nul byte")
    }
}

#[stable(feature = "frombyteswithnulerror_impls", since = "1.17.0")]
impl Error for FromBytesWithNulError {
    fn description(&self) -> &str {
        match self.kind {
            FromBytesWithNulErrorKind::InteriorNul(..) =>
                "data provided contains an interior nul byte",
            FromBytesWithNulErrorKind::NotNulTerminated =>
                "data provided is not nul terminated",
        }
    }
}

#[stable(feature = "frombyteswithnulerror_impls", since = "1.17.0")]
impl fmt::Display for FromBytesWithNulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())?;
        if let FromBytesWithNulErrorKind::InteriorNul(pos) = self.kind {
            write!(f, " at byte pos {}", pos)?;
        }
        Ok(())
    }
}

impl IntoStringError {
    /// Consumes this error, returning original [`CString`] which generated the
    /// error.
    ///
    /// [`CString`]: struct.CString.html
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_cstring(self) -> CString {
        self.inner
    }

    /// Access the underlying UTF-8 error that was the cause of this error.
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn utf8_error(&self) -> Utf8Error {
        self.error
    }
}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl Error for IntoStringError {
    fn description(&self) -> &str {
        "C string contained non-utf8 bytes"
    }

    fn cause(&self) -> Option<&dyn Error> {
        Some(&self.error)
    }
}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl fmt::Display for IntoStringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.description().fmt(f)
    }
}

impl CStr {
    /// Wraps a raw C string with a safe C string wrapper.
    ///
    /// This function will wrap the provided `ptr` with a `CStr` wrapper, which
    /// allows inspection and interoperation of non-owned C strings. This method
    /// is unsafe for a number of reasons:
    ///
    /// * There is no guarantee to the validity of `ptr`.
    /// * The returned lifetime is not guaranteed to be the actual lifetime of
    ///   `ptr`.
    /// * There is no guarantee that the memory pointed to by `ptr` contains a
    ///   valid nul terminator byte at the end of the string.
    /// * It is not guaranteed that the memory pointed by `ptr` won't change
    ///   before the `CStr` has been destroyed.
    ///
    /// > **Note**: This operation is intended to be a 0-cost cast but it is
    /// > currently implemented with an up-front calculation of the length of
    /// > the string. This is not guaranteed to always be the case.
    ///
    /// # Examples
    ///
    /// ```ignore (extern-declaration)
    /// # fn main() {
    /// use std::ffi::CStr;
    /// use std::os::raw::c_char;
    ///
    /// extern {
    ///     fn my_string() -> *const c_char;
    /// }
    ///
    /// unsafe {
    ///     let slice = CStr::from_ptr(my_string());
    ///     println!("string returned: {}", slice.to_str().unwrap());
    /// }
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_ptr<'a>(ptr: *const c_char) -> &'a CStr {
        let len = sys::strlen(ptr);
        let ptr = ptr as *const u8;
        CStr::from_bytes_with_nul_unchecked(slice::from_raw_parts(ptr, len as usize + 1))
    }

    /// Creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CStr`
    /// wrapper after ensuring that the byte slice is nul-terminated
    /// and does not contain any interior nul bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let cstr = CStr::from_bytes_with_nul(b"hello\0");
    /// assert!(cstr.is_ok());
    /// ```
    ///
    /// Creating a `CStr` without a trailing nul terminator is an error:
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"hello");
    /// assert!(c_str.is_err());
    /// ```
    ///
    /// Creating a `CStr` with an interior nul byte is an error:
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"he\0llo\0");
    /// assert!(c_str.is_err());
    /// ```
    #[stable(feature = "cstr_from_bytes", since = "1.10.0")]
    pub fn from_bytes_with_nul(bytes: &[u8])
                               -> Result<&CStr, FromBytesWithNulError> {
        let nul_pos = memchr::memchr(0, bytes);
        if let Some(nul_pos) = nul_pos {
            if nul_pos + 1 != bytes.len() {
                return Err(FromBytesWithNulError::interior_nul(nul_pos));
            }
            Ok(unsafe { CStr::from_bytes_with_nul_unchecked(bytes) })
        } else {
            Err(FromBytesWithNulError::not_nul_terminated())
        }
    }

    /// Unsafely creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CStr` wrapper without
    /// performing any sanity checks. The provided slice **must** be nul-terminated
    /// and not contain any interior nul bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{CStr, CString};
    ///
    /// unsafe {
    ///     let cstring = CString::new("hello").expect("CString::new failed");
    ///     let cstr = CStr::from_bytes_with_nul_unchecked(cstring.to_bytes_with_nul());
    ///     assert_eq!(cstr, &*cstring);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "cstr_from_bytes", since = "1.10.0")]
    #[rustc_const_unstable(feature = "const_cstr_unchecked")]
    pub const unsafe fn from_bytes_with_nul_unchecked(bytes: &[u8]) -> &CStr {
        &*(bytes as *const [u8] as *const CStr)
    }

    /// Returns the inner pointer to this C string.
    ///
    /// The returned pointer will be valid for as long as `self` is, and points
    /// to a contiguous region of memory terminated with a 0 byte to represent
    /// the end of the string.
    ///
    /// **WARNING**
    ///
    /// The returned pointer is read-only; writing to it (including passing it
    /// to C code that writes to it) causes undefined behavior.
    ///
    /// It is your responsibility to make sure that the underlying memory is not
    /// freed too early. For example, the following code will cause undefined
    /// behavior when `ptr` is used inside the `unsafe` block:
    ///
    /// ```no_run
    /// # #![allow(unused_must_use)]
    /// use std::ffi::{CString};
    ///
    /// let ptr = CString::new("Hello").expect("CString::new failed").as_ptr();
    /// unsafe {
    ///     // `ptr` is dangling
    ///     *ptr;
    /// }
    /// ```
    ///
    /// This happens because the pointer returned by `as_ptr` does not carry any
    /// lifetime information and the [`CString`] is deallocated immediately after
    /// the `CString::new("Hello").expect("CString::new failed").as_ptr()` expression is evaluated.
    /// To fix the problem, bind the `CString` to a local variable:
    ///
    /// ```no_run
    /// # #![allow(unused_must_use)]
    /// use std::ffi::{CString};
    ///
    /// let hello = CString::new("Hello").expect("CString::new failed");
    /// let ptr = hello.as_ptr();
    /// unsafe {
    ///     // `ptr` is valid because `hello` is in scope
    ///     *ptr;
    /// }
    /// ```
    ///
    /// This way, the lifetime of the `CString` in `hello` encompasses
    /// the lifetime of `ptr` and the `unsafe` block.
    ///
    /// [`CString`]: struct.CString.html
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn as_ptr(&self) -> *const c_char {
        self.inner.as_ptr()
    }

    /// Converts this C string to a byte slice.
    ///
    /// The returned slice will **not** contain the trailing nul terminator that this C
    /// string has.
    ///
    /// > **Note**: This method is currently implemented as a constant-time
    /// > cast, but it is planned to alter its definition in the future to
    /// > perform the length calculation whenever this method is called.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(c_str.to_bytes(), b"foo");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_bytes(&self) -> &[u8] {
        let bytes = self.to_bytes_with_nul();
        &bytes[..bytes.len() - 1]
    }

    /// Converts this C string to a byte slice containing the trailing 0 byte.
    ///
    /// This function is the equivalent of [`to_bytes`] except that it will retain
    /// the trailing nul terminator instead of chopping it off.
    ///
    /// > **Note**: This method is currently implemented as a 0-cost cast, but
    /// > it is planned to alter its definition in the future to perform the
    /// > length calculation whenever this method is called.
    ///
    /// [`to_bytes`]: #method.to_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(c_str.to_bytes_with_nul(), b"foo\0");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_bytes_with_nul(&self) -> &[u8] {
        unsafe { &*(&self.inner as *const [c_char] as *const [u8]) }
    }

    /// Yields a [`&str`] slice if the `CStr` contains valid UTF-8.
    ///
    /// If the contents of the `CStr` are valid UTF-8 data, this
    /// function will return the corresponding [`&str`] slice. Otherwise,
    /// it will return an error with details of where UTF-8 validation failed.
    ///
    /// > **Note**: This method is currently implemented to check for validity
    /// > after a constant-time cast, but it is planned to alter its definition
    /// > in the future to perform the length calculation in addition to the
    /// > UTF-8 check whenever this method is called.
    ///
    /// [`&str`]: ../primitive.str.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(c_str.to_str(), Ok("foo"));
    /// ```
    #[stable(feature = "cstr_to_str", since = "1.4.0")]
    pub fn to_str(&self) -> Result<&str, str::Utf8Error> {
        // N.B., when `CStr` is changed to perform the length check in `.to_bytes()`
        // instead of in `from_ptr()`, it may be worth considering if this should
        // be rewritten to do the UTF-8 check inline with the length calculation
        // instead of doing it afterwards.
        str::from_utf8(self.to_bytes())
    }

    /// Converts a `CStr` into a [`Cow`]`<`[`str`]`>`.
    ///
    /// If the contents of the `CStr` are valid UTF-8 data, this
    /// function will return a [`Cow`]`::`[`Borrowed`]`(`[`&str`]`)`
    /// with the corresponding [`&str`] slice. Otherwise, it will
    /// replace any invalid UTF-8 sequences with
    /// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD] and return a
    /// [`Cow`]`::`[`Owned`]`(`[`String`]`)` with the result.
    ///
    /// > **Note**: This method is currently implemented to check for validity
    /// > after a constant-time cast, but it is planned to alter its definition
    /// > in the future to perform the length calculation in addition to the
    /// > UTF-8 check whenever this method is called.
    ///
    /// [`Cow`]: ../borrow/enum.Cow.html
    /// [`Borrowed`]: ../borrow/enum.Cow.html#variant.Borrowed
    /// [`Owned`]: ../borrow/enum.Cow.html#variant.Owned
    /// [`str`]: ../primitive.str.html
    /// [`String`]: ../string/struct.String.html
    /// [U+FFFD]: ../char/constant.REPLACEMENT_CHARACTER.html
    ///
    /// # Examples
    ///
    /// Calling `to_string_lossy` on a `CStr` containing valid UTF-8:
    ///
    /// ```
    /// use std::borrow::Cow;
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"Hello World\0")
    ///                  .expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(c_str.to_string_lossy(), Cow::Borrowed("Hello World"));
    /// ```
    ///
    /// Calling `to_string_lossy` on a `CStr` containing invalid UTF-8:
    ///
    /// ```
    /// use std::borrow::Cow;
    /// use std::ffi::CStr;
    ///
    /// let c_str = CStr::from_bytes_with_nul(b"Hello \xF0\x90\x80World\0")
    ///                  .expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(
    ///     c_str.to_string_lossy(),
    ///     Cow::Owned(String::from("Hello �World")) as Cow<'_, str>
    /// );
    /// ```
    #[stable(feature = "cstr_to_str", since = "1.4.0")]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.to_bytes())
    }

    /// Converts a [`Box`]`<CStr>` into a [`CString`] without copying or allocating.
    ///
    /// [`Box`]: ../boxed/struct.Box.html
    /// [`CString`]: struct.CString.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::new(b"foo".to_vec()).expect("CString::new failed");
    /// let boxed = c_string.into_boxed_c_str();
    /// assert_eq!(boxed.into_c_string(), CString::new("foo").expect("CString::new failed"));
    /// ```
    #[stable(feature = "into_boxed_c_str", since = "1.20.0")]
    pub fn into_c_string(self: Box<CStr>) -> CString {
        let raw = Box::into_raw(self) as *mut [u8];
        CString { inner: unsafe { Box::from_raw(raw) } }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for CStr {
    fn eq(&self, other: &CStr) -> bool {
        self.to_bytes().eq(other.to_bytes())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for CStr {}
#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for CStr {
    fn partial_cmp(&self, other: &CStr) -> Option<Ordering> {
        self.to_bytes().partial_cmp(&other.to_bytes())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for CStr {
    fn cmp(&self, other: &CStr) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

#[stable(feature = "cstr_borrow", since = "1.3.0")]
impl ToOwned for CStr {
    type Owned = CString;

    fn to_owned(&self) -> CString {
        CString { inner: self.to_bytes_with_nul().into() }
    }
}

#[stable(feature = "cstring_asref", since = "1.7.0")]
impl From<&CStr> for CString {
    fn from(s: &CStr) -> CString {
        s.to_owned()
    }
}

#[stable(feature = "cstring_asref", since = "1.7.0")]
impl ops::Index<ops::RangeFull> for CString {
    type Output = CStr;

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &CStr {
        self
    }
}

#[stable(feature = "cstring_asref", since = "1.7.0")]
impl AsRef<CStr> for CStr {
    #[inline]
    fn as_ref(&self) -> &CStr {
        self
    }
}

#[stable(feature = "cstring_asref", since = "1.7.0")]
impl AsRef<CStr> for CString {
    #[inline]
    fn as_ref(&self) -> &CStr {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::raw::c_char;
    use crate::borrow::Cow::{Borrowed, Owned};
    use crate::hash::{Hash, Hasher};
    use crate::collections::hash_map::DefaultHasher;
    use crate::rc::Rc;
    use crate::sync::Arc;

    #[test]
    fn c_to_rust() {
        let data = b"123\0";
        let ptr = data.as_ptr() as *const c_char;
        unsafe {
            assert_eq!(CStr::from_ptr(ptr).to_bytes(), b"123");
            assert_eq!(CStr::from_ptr(ptr).to_bytes_with_nul(), b"123\0");
        }
    }

    #[test]
    fn simple() {
        let s = CString::new("1234").unwrap();
        assert_eq!(s.as_bytes(), b"1234");
        assert_eq!(s.as_bytes_with_nul(), b"1234\0");
    }

    #[test]
    fn build_with_zero1() {
        assert!(CString::new(&b"\0"[..]).is_err());
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
        let s = CString::new(&b"abc\x01\x02\n\xE2\x80\xA6\xFF"[..]).unwrap();
        assert_eq!(format!("{:?}", s), r#""abc\x01\x02\n\xe2\x80\xa6\xff""#);
    }

    #[test]
    fn borrowed() {
        unsafe {
            let s = CStr::from_ptr(b"12\0".as_ptr() as *const _);
            assert_eq!(s.to_bytes(), b"12");
            assert_eq!(s.to_bytes_with_nul(), b"12\0");
        }
    }

    #[test]
    fn to_str() {
        let data = b"123\xE2\x80\xA6\0";
        let ptr = data.as_ptr() as *const c_char;
        unsafe {
            assert_eq!(CStr::from_ptr(ptr).to_str(), Ok("123…"));
            assert_eq!(CStr::from_ptr(ptr).to_string_lossy(), Borrowed("123…"));
        }
        let data = b"123\xE2\0";
        let ptr = data.as_ptr() as *const c_char;
        unsafe {
            assert!(CStr::from_ptr(ptr).to_str().is_err());
            assert_eq!(CStr::from_ptr(ptr).to_string_lossy(), Owned::<str>(format!("123\u{FFFD}")));
        }
    }

    #[test]
    fn to_owned() {
        let data = b"123\0";
        let ptr = data.as_ptr() as *const c_char;

        let owned = unsafe { CStr::from_ptr(ptr).to_owned() };
        assert_eq!(owned.as_bytes_with_nul(), data);
    }

    #[test]
    fn equal_hash() {
        let data = b"123\xE2\xFA\xA6\0";
        let ptr = data.as_ptr() as *const c_char;
        let cstr: &'static CStr = unsafe { CStr::from_ptr(ptr) };

        let mut s = DefaultHasher::new();
        cstr.hash(&mut s);
        let cstr_hash = s.finish();
        let mut s = DefaultHasher::new();
        CString::new(&data[..data.len() - 1]).unwrap().hash(&mut s);
        let cstring_hash = s.finish();

        assert_eq!(cstr_hash, cstring_hash);
    }

    #[test]
    fn from_bytes_with_nul() {
        let data = b"123\0";
        let cstr = CStr::from_bytes_with_nul(data);
        assert_eq!(cstr.map(CStr::to_bytes), Ok(&b"123"[..]));
        let cstr = CStr::from_bytes_with_nul(data);
        assert_eq!(cstr.map(CStr::to_bytes_with_nul), Ok(&b"123\0"[..]));

        unsafe {
            let cstr = CStr::from_bytes_with_nul(data);
            let cstr_unchecked = CStr::from_bytes_with_nul_unchecked(data);
            assert_eq!(cstr, Ok(cstr_unchecked));
        }
    }

    #[test]
    fn from_bytes_with_nul_unterminated() {
        let data = b"123";
        let cstr = CStr::from_bytes_with_nul(data);
        assert!(cstr.is_err());
    }

    #[test]
    fn from_bytes_with_nul_interior() {
        let data = b"1\023\0";
        let cstr = CStr::from_bytes_with_nul(data);
        assert!(cstr.is_err());
    }

    #[test]
    fn into_boxed() {
        let orig: &[u8] = b"Hello, world!\0";
        let cstr = CStr::from_bytes_with_nul(orig).unwrap();
        let boxed: Box<CStr> = Box::from(cstr);
        let cstring = cstr.to_owned().into_boxed_c_str().into_c_string();
        assert_eq!(cstr, &*boxed);
        assert_eq!(&*boxed, &*cstring);
        assert_eq!(&*cstring, cstr);
    }

    #[test]
    fn boxed_default() {
        let boxed = <Box<CStr>>::default();
        assert_eq!(boxed.to_bytes_with_nul(), &[0]);
    }

    #[test]
    fn into_rc() {
        let orig: &[u8] = b"Hello, world!\0";
        let cstr = CStr::from_bytes_with_nul(orig).unwrap();
        let rc: Rc<CStr> = Rc::from(cstr);
        let arc: Arc<CStr> = Arc::from(cstr);

        assert_eq!(&*rc, cstr);
        assert_eq!(&*arc, cstr);

        let rc2: Rc<CStr> = Rc::from(cstr.to_owned());
        let arc2: Arc<CStr> = Arc::from(cstr.to_owned());

        assert_eq!(&*rc2, cstr);
        assert_eq!(&*arc2, cstr);
    }

    #[test]
    fn cstr_const_constructor() {
        const CSTR: &CStr = unsafe {
            CStr::from_bytes_with_nul_unchecked(b"Hello, world!\0")
        };

        assert_eq!(CSTR.to_str().unwrap(), "Hello, world!");
    }
}
