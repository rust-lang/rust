//! [`CString`] and its related types.

use core::borrow::Borrow;
use core::ffi::{CStr, c_char};
use core::num::NonZero;
use core::slice::memchr;
use core::str::{self, FromStr, Utf8Error};
use core::{fmt, mem, ops, ptr, slice};

use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
use crate::rc::Rc;
use crate::string::String;
#[cfg(target_has_atomic = "ptr")]
use crate::sync::Arc;
use crate::vec::Vec;

/// A type representing an owned, C-compatible, nul-terminated string with no nul bytes in the
/// middle.
///
/// This type serves the purpose of being able to safely generate a
/// C-compatible string from a Rust byte slice or vector. An instance of this
/// type is a static guarantee that the underlying bytes contain no interior 0
/// bytes ("nul characters") and that the final byte is 0 ("nul terminator").
///
/// `CString` is to <code>&[CStr]</code> as [`String`] is to <code>&[str]</code>: the former
/// in each pair are owned strings; the latter are borrowed
/// references.
///
/// # Creating a `CString`
///
/// A `CString` is created from either a byte slice or a byte vector,
/// or anything that implements <code>[Into]<[Vec]<[u8]>></code> (for
/// example, you can build a `CString` straight out of a [`String`] or
/// a <code>&[str]</code>, since both implement that trait).
/// You can create a `CString` from a literal with `CString::from(c"Text")`.
///
/// The [`CString::new`] method will actually check that the provided <code>&[[u8]]</code>
/// does not have 0 bytes in the middle, and return an error if it
/// finds one.
///
/// # Extracting a raw pointer to the whole C string
///
/// `CString` implements an [`as_ptr`][`CStr::as_ptr`] method through the [`Deref`]
/// trait. This method will give you a `*const c_char` which you can
/// feed directly to extern functions that expect a nul-terminated
/// string, like C's `strdup()`. Notice that [`as_ptr`][`CStr::as_ptr`] returns a
/// read-only pointer; if the C code writes to it, that causes
/// undefined behavior.
///
/// # Extracting a slice of the whole C string
///
/// Alternatively, you can obtain a <code>&[[u8]]</code> slice from a
/// `CString` with the [`CString::as_bytes`] method. Slices produced in this
/// way do *not* contain the trailing nul terminator. This is useful
/// when you will be calling an extern function that takes a `*const
/// u8` argument which is not necessarily nul-terminated, plus another
/// argument with the length of the string — like C's `strndup()`.
/// You can of course get the slice's length with its
/// [`len`][slice::len] method.
///
/// If you need a <code>&[[u8]]</code> slice *with* the nul terminator, you
/// can use [`CString::as_bytes_with_nul`] instead.
///
/// Once you have the kind of slice you need (with or without a nul
/// terminator), you can call the slice's own
/// [`as_ptr`][slice::as_ptr] method to get a read-only raw pointer to pass to
/// extern functions. See the documentation for that function for a
/// discussion on ensuring the lifetime of the raw pointer.
///
/// [str]: prim@str "str"
/// [`Deref`]: ops::Deref
///
/// # Examples
///
/// ```ignore (extern-declaration)
/// # fn main() {
/// use std::ffi::CString;
/// use std::os::raw::c_char;
///
/// extern "C" {
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
#[rustc_diagnostic_item = "cstring_type"]
#[stable(feature = "alloc_c_string", since = "1.64.0")]
pub struct CString {
    // Invariant 1: the slice ends with a zero byte and has a length of at least one.
    // Invariant 2: the slice contains only one zero byte.
    // Improper usage of unsafe function can break Invariant 2, but not Invariant 1.
    inner: Box<[u8]>,
}

/// An error indicating that an interior nul byte was found.
///
/// While Rust strings may contain nul bytes in the middle, C strings
/// can't, as that byte would effectively truncate the string.
///
/// This error is created by the [`new`][`CString::new`] method on
/// [`CString`]. See its documentation for more.
///
/// # Examples
///
/// ```
/// use std::ffi::{CString, NulError};
///
/// let _: NulError = CString::new(b"f\0oo".to_vec()).unwrap_err();
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "alloc_c_string", since = "1.64.0")]
pub struct NulError(usize, Vec<u8>);

#[derive(Clone, PartialEq, Eq, Debug)]
enum FromBytesWithNulErrorKind {
    InteriorNul(usize),
    NotNulTerminated,
}

/// An error indicating that a nul byte was not in the expected position.
///
/// The vector used to create a [`CString`] must have one and only one nul byte,
/// positioned at the end.
///
/// This error is created by the [`CString::from_vec_with_nul`] method.
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use std::ffi::{CString, FromVecWithNulError};
///
/// let _: FromVecWithNulError = CString::from_vec_with_nul(b"f\0oo".to_vec()).unwrap_err();
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "alloc_c_string", since = "1.64.0")]
pub struct FromVecWithNulError {
    error_kind: FromBytesWithNulErrorKind,
    bytes: Vec<u8>,
}

#[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
impl FromVecWithNulError {
    /// Returns a slice of [`u8`]s bytes that were attempted to convert to a [`CString`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// // Some invalid bytes in a vector
    /// let bytes = b"f\0oo".to_vec();
    ///
    /// let value = CString::from_vec_with_nul(bytes.clone());
    ///
    /// assert_eq!(&bytes[..], value.unwrap_err().as_bytes());
    /// ```
    #[must_use]
    #[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..]
    }

    /// Returns the bytes that were attempted to convert to a [`CString`].
    ///
    /// This method is carefully constructed to avoid allocation. It will
    /// consume the error, moving out the bytes, so that a copy of the bytes
    /// does not need to be made.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// // Some invalid bytes in a vector
    /// let bytes = b"f\0oo".to_vec();
    ///
    /// let value = CString::from_vec_with_nul(bytes.clone());
    ///
    /// assert_eq!(bytes, value.unwrap_err().into_bytes());
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// An error indicating invalid UTF-8 when converting a [`CString`] into a [`String`].
///
/// `CString` is just a wrapper over a buffer of bytes with a nul terminator;
/// [`CString::into_string`] performs UTF-8 validation on those bytes and may
/// return this error.
///
/// This `struct` is created by [`CString::into_string()`]. See
/// its documentation for more.
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "alloc_c_string", since = "1.64.0")]
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
    /// extern "C" { fn puts(s: *const c_char); }
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<T: Into<Vec<u8>>>(t: T) -> Result<CString, NulError> {
        trait SpecNewImpl {
            fn spec_new_impl(self) -> Result<CString, NulError>;
        }

        impl<T: Into<Vec<u8>>> SpecNewImpl for T {
            default fn spec_new_impl(self) -> Result<CString, NulError> {
                let bytes: Vec<u8> = self.into();
                match memchr::memchr(0, &bytes) {
                    Some(i) => Err(NulError(i, bytes)),
                    None => Ok(unsafe { CString::_from_vec_unchecked(bytes) }),
                }
            }
        }

        // Specialization for avoiding reallocation
        #[inline(always)] // Without that it is not inlined into specializations
        fn spec_new_impl_bytes(bytes: &[u8]) -> Result<CString, NulError> {
            // We cannot have such large slice that we would overflow here
            // but using `checked_add` allows LLVM to assume that capacity never overflows
            // and generate twice shorter code.
            // `saturating_add` doesn't help for some reason.
            let capacity = bytes.len().checked_add(1).unwrap();

            // Allocate before validation to avoid duplication of allocation code.
            // We still need to allocate and copy memory even if we get an error.
            let mut buffer = Vec::with_capacity(capacity);
            buffer.extend(bytes);

            // Check memory of self instead of new buffer.
            // This allows better optimizations if lto enabled.
            match memchr::memchr(0, bytes) {
                Some(i) => Err(NulError(i, buffer)),
                None => Ok(unsafe { CString::_from_vec_unchecked(buffer) }),
            }
        }

        impl SpecNewImpl for &'_ [u8] {
            fn spec_new_impl(self) -> Result<CString, NulError> {
                spec_new_impl_bytes(self)
            }
        }

        impl SpecNewImpl for &'_ str {
            fn spec_new_impl(self) -> Result<CString, NulError> {
                spec_new_impl_bytes(self.as_bytes())
            }
        }

        impl SpecNewImpl for &'_ mut [u8] {
            fn spec_new_impl(self) -> Result<CString, NulError> {
                spec_new_impl_bytes(self)
            }
        }

        t.spec_new_impl()
    }

    /// Creates a C-compatible string by consuming a byte vector,
    /// without checking for interior 0 bytes.
    ///
    /// Trailing 0 byte will be appended by this function.
    ///
    /// This method is equivalent to [`CString::new`] except that no runtime
    /// assertion is made that `v` contains no 0 bytes, and it requires an
    /// actual byte vector, not anything that can be converted to one with Into.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_vec_unchecked(v: Vec<u8>) -> Self {
        debug_assert!(memchr::memchr(0, &v).is_none());
        unsafe { Self::_from_vec_unchecked(v) }
    }

    unsafe fn _from_vec_unchecked(mut v: Vec<u8>) -> Self {
        v.reserve_exact(1);
        v.push(0);
        Self { inner: v.into_boxed_slice() }
    }

    /// Retakes ownership of a `CString` that was transferred to C via
    /// [`CString::into_raw`].
    ///
    /// Additionally, the length of the string will be recalculated from the pointer.
    ///
    /// # Safety
    ///
    /// This should only ever be called with a pointer that was earlier
    /// obtained by calling [`CString::into_raw`], and the memory it points to must not be accessed
    /// through any other pointer during the lifetime of reconstructed `CString`.
    /// Other usage (e.g., trying to take ownership of a string that was allocated by foreign code)
    /// is likely to lead to undefined behavior or allocator corruption.
    ///
    /// This function does not validate ownership of the raw pointer's memory.
    /// A double-free may occur if the function is called twice on the same raw pointer.
    /// Additionally, the caller must ensure the pointer is not dangling.
    ///
    /// It should be noted that the length isn't just "recomputed," but that
    /// the recomputed length must match the original length from the
    /// [`CString::into_raw`] call. This means the [`CString::into_raw`]/`from_raw`
    /// methods should not be used when passing the string to C functions that can
    /// modify the string's length.
    ///
    /// > **Note:** If you need to borrow a string that was allocated by
    /// > foreign code, use [`CStr`]. If you need to take ownership of
    /// > a string that was allocated by foreign code, you will need to
    /// > make your own provisions for freeing it appropriately, likely
    /// > with the foreign code's API to do that.
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
    /// extern "C" {
    ///     fn some_extern_function(s: *mut c_char);
    /// }
    ///
    /// let c_string = CString::from(c"Hello!");
    /// let raw = c_string.into_raw();
    /// unsafe {
    ///     some_extern_function(raw);
    ///     let c_string = CString::from_raw(raw);
    /// }
    /// ```
    #[must_use = "call `drop(from_raw(ptr))` if you intend to drop the `CString`"]
    #[stable(feature = "cstr_memory", since = "1.4.0")]
    pub unsafe fn from_raw(ptr: *mut c_char) -> CString {
        // SAFETY: This is called with a pointer that was obtained from a call
        // to `CString::into_raw` and the length has not been modified. As such,
        // we know there is a NUL byte (and only one) at the end and that the
        // information about the size of the allocation is correct on Rust's
        // side.
        unsafe {
            unsafe extern "C" {
                /// Provided by libc or compiler_builtins.
                fn strlen(s: *const c_char) -> usize;
            }
            let len = strlen(ptr) + 1; // Including the NUL byte
            let slice = slice::from_raw_parts_mut(ptr, len);
            CString { inner: Box::from_raw(slice as *mut [c_char] as *mut [u8]) }
        }
    }

    /// Consumes the `CString` and transfers ownership of the string to a C caller.
    ///
    /// The pointer which this function returns must be returned to Rust and reconstituted using
    /// [`CString::from_raw`] to be properly deallocated. Specifically, one
    /// should *not* use the standard C `free()` function to deallocate
    /// this string.
    ///
    /// Failure to call [`CString::from_raw`] will lead to a memory leak.
    ///
    /// The C side must **not** modify the length of the string (by writing a
    /// nul byte somewhere inside the string or removing the final one) before
    /// it makes it back into Rust using [`CString::from_raw`]. See the safety section
    /// in [`CString::from_raw`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::from(c"foo");
    ///
    /// let ptr = c_string.into_raw();
    ///
    /// unsafe {
    ///     assert_eq!(b'f', *ptr as u8);
    ///     assert_eq!(b'o', *ptr.add(1) as u8);
    ///     assert_eq!(b'o', *ptr.add(2) as u8);
    ///     assert_eq!(b'\0', *ptr.add(3) as u8);
    ///
    ///     // retake pointer to free memory
    ///     let _ = CString::from_raw(ptr);
    /// }
    /// ```
    #[inline]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "cstr_memory", since = "1.4.0")]
    pub fn into_raw(self) -> *mut c_char {
        Box::into_raw(self.into_inner()) as *mut c_char
    }

    /// Converts the `CString` into a [`String`] if it contains valid UTF-8 data.
    ///
    /// On failure, ownership of the original `CString` is returned.
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
        String::from_utf8(self.into_bytes()).map_err(|e| IntoStringError {
            error: e.utf8_error(),
            inner: unsafe { Self::_from_vec_unchecked(e.into_bytes()) },
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
    /// let c_string = CString::from(c"foo");
    /// let bytes = c_string.into_bytes();
    /// assert_eq!(bytes, vec![b'f', b'o', b'o']);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_bytes(self) -> Vec<u8> {
        let mut vec = self.into_inner().into_vec();
        let _nul = vec.pop();
        debug_assert_eq!(_nul, Some(0u8));
        vec
    }

    /// Equivalent to [`CString::into_bytes()`] except that the
    /// returned vector includes the trailing nul terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::from(c"foo");
    /// let bytes = c_string.into_bytes_with_nul();
    /// assert_eq!(bytes, vec![b'f', b'o', b'o', b'\0']);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_bytes_with_nul(self) -> Vec<u8> {
        self.into_inner().into_vec()
    }

    /// Returns the contents of this `CString` as a slice of bytes.
    ///
    /// The returned slice does **not** contain the trailing nul
    /// terminator, and it is guaranteed to not have any interior nul
    /// bytes. If you need the nul terminator, use
    /// [`CString::as_bytes_with_nul`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::from(c"foo");
    /// let bytes = c_string.as_bytes();
    /// assert_eq!(bytes, &[b'f', b'o', b'o']);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: CString has a length at least 1
        unsafe { self.inner.get_unchecked(..self.inner.len() - 1) }
    }

    /// Equivalent to [`CString::as_bytes()`] except that the
    /// returned slice includes the trailing nul terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CString;
    ///
    /// let c_string = CString::from(c"foo");
    /// let bytes = c_string.as_bytes_with_nul();
    /// assert_eq!(bytes, &[b'f', b'o', b'o', b'\0']);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        &self.inner
    }

    /// Extracts a [`CStr`] slice containing the entire string.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{CString, CStr};
    ///
    /// let c_string = CString::from(c"foo");
    /// let cstr = c_string.as_c_str();
    /// assert_eq!(cstr,
    ///            CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed"));
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "as_c_str", since = "1.20.0")]
    #[rustc_diagnostic_item = "cstring_as_c_str"]
    pub fn as_c_str(&self) -> &CStr {
        unsafe { CStr::from_bytes_with_nul_unchecked(self.as_bytes_with_nul()) }
    }

    /// Converts this `CString` into a boxed [`CStr`].
    ///
    /// # Examples
    ///
    /// ```
    /// let c_string = c"foo".to_owned();
    /// let boxed = c_string.into_boxed_c_str();
    /// assert_eq!(boxed.to_bytes_with_nul(), b"foo\0");
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "into_boxed_c_str", since = "1.20.0")]
    pub fn into_boxed_c_str(self) -> Box<CStr> {
        unsafe { Box::from_raw(Box::into_raw(self.into_inner()) as *mut CStr) }
    }

    /// Bypass "move out of struct which implements [`Drop`] trait" restriction.
    #[inline]
    fn into_inner(self) -> Box<[u8]> {
        // Rationale: `mem::forget(self)` invalidates the previous call to `ptr::read(&self.inner)`
        // so we use `ManuallyDrop` to ensure `self` is not dropped.
        // Then we can return the box directly without invalidating it.
        // See https://github.com/rust-lang/rust/issues/62553.
        let this = mem::ManuallyDrop::new(self);
        unsafe { ptr::read(&this.inner) }
    }

    /// Converts a <code>[Vec]<[u8]></code> to a [`CString`] without checking the
    /// invariants on the given [`Vec`].
    ///
    /// # Safety
    ///
    /// The given [`Vec`] **must** have one nul byte as its last element.
    /// This means it cannot be empty nor have any other nul byte anywhere else.
    ///
    /// # Example
    ///
    /// ```
    /// use std::ffi::CString;
    /// assert_eq!(
    ///     unsafe { CString::from_vec_with_nul_unchecked(b"abc\0".to_vec()) },
    ///     unsafe { CString::from_vec_unchecked(b"abc".to_vec()) }
    /// );
    /// ```
    #[must_use]
    #[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
    pub unsafe fn from_vec_with_nul_unchecked(v: Vec<u8>) -> Self {
        debug_assert!(memchr::memchr(0, &v).unwrap() + 1 == v.len());
        unsafe { Self::_from_vec_with_nul_unchecked(v) }
    }

    unsafe fn _from_vec_with_nul_unchecked(v: Vec<u8>) -> Self {
        Self { inner: v.into_boxed_slice() }
    }

    /// Attempts to converts a <code>[Vec]<[u8]></code> to a [`CString`].
    ///
    /// Runtime checks are present to ensure there is only one nul byte in the
    /// [`Vec`], its last element.
    ///
    /// # Errors
    ///
    /// If a nul byte is present and not the last element or no nul bytes
    /// is present, an error will be returned.
    ///
    /// # Examples
    ///
    /// A successful conversion will produce the same result as [`CString::new`]
    /// when called without the ending nul byte.
    ///
    /// ```
    /// use std::ffi::CString;
    /// assert_eq!(
    ///     CString::from_vec_with_nul(b"abc\0".to_vec())
    ///         .expect("CString::from_vec_with_nul failed"),
    ///     c"abc".to_owned()
    /// );
    /// ```
    ///
    /// An incorrectly formatted [`Vec`] will produce an error.
    ///
    /// ```
    /// use std::ffi::{CString, FromVecWithNulError};
    /// // Interior nul byte
    /// let _: FromVecWithNulError = CString::from_vec_with_nul(b"a\0bc".to_vec()).unwrap_err();
    /// // No nul byte
    /// let _: FromVecWithNulError = CString::from_vec_with_nul(b"abc".to_vec()).unwrap_err();
    /// ```
    #[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
    pub fn from_vec_with_nul(v: Vec<u8>) -> Result<Self, FromVecWithNulError> {
        let nul_pos = memchr::memchr(0, &v);
        match nul_pos {
            Some(nul_pos) if nul_pos + 1 == v.len() => {
                // SAFETY: We know there is only one nul byte, at the end
                // of the vec.
                Ok(unsafe { Self::_from_vec_with_nul_unchecked(v) })
            }
            Some(nul_pos) => Err(FromVecWithNulError {
                error_kind: FromBytesWithNulErrorKind::InteriorNul(nul_pos),
                bytes: v,
            }),
            None => Err(FromVecWithNulError {
                error_kind: FromBytesWithNulErrorKind::NotNulTerminated,
                bytes: v,
            }),
        }
    }
}

// Turns this `CString` into an empty string to prevent
// memory-unsafe code from working by accident. Inline
// to prevent LLVM from optimizing it away in debug builds.
#[stable(feature = "cstring_drop", since = "1.13.0")]
#[rustc_insignificant_dtor]
impl Drop for CString {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            *self.inner.get_unchecked_mut(0) = 0;
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for CString {
    type Target = CStr;

    #[inline]
    fn deref(&self) -> &CStr {
        self.as_c_str()
    }
}

/// Delegates to the [`CStr`] implementation of [`fmt::Debug`],
/// showing invalid UTF-8 as hex escapes.
#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for CString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_c_str(), f)
    }
}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl From<CString> for Vec<u8> {
    /// Converts a [`CString`] into a <code>[Vec]<[u8]></code>.
    ///
    /// The conversion consumes the [`CString`], and removes the terminating NUL byte.
    #[inline]
    fn from(s: CString) -> Vec<u8> {
        s.into_bytes()
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
    fn borrow(&self) -> &CStr {
        self
    }
}

#[stable(feature = "cstring_from_cow_cstr", since = "1.28.0")]
impl<'a> From<Cow<'a, CStr>> for CString {
    /// Converts a `Cow<'a, CStr>` into a `CString`, by copying the contents if they are
    /// borrowed.
    #[inline]
    fn from(s: Cow<'a, CStr>) -> Self {
        s.into_owned()
    }
}

#[stable(feature = "box_from_c_str", since = "1.17.0")]
impl From<&CStr> for Box<CStr> {
    /// Converts a `&CStr` into a `Box<CStr>`,
    /// by copying the contents into a newly allocated [`Box`].
    fn from(s: &CStr) -> Box<CStr> {
        let boxed: Box<[u8]> = Box::from(s.to_bytes_with_nul());
        unsafe { Box::from_raw(Box::into_raw(boxed) as *mut CStr) }
    }
}

#[stable(feature = "box_from_mut_slice", since = "1.84.0")]
impl From<&mut CStr> for Box<CStr> {
    /// Converts a `&mut CStr` into a `Box<CStr>`,
    /// by copying the contents into a newly allocated [`Box`].
    fn from(s: &mut CStr) -> Box<CStr> {
        Self::from(&*s)
    }
}

#[stable(feature = "box_from_cow", since = "1.45.0")]
impl From<Cow<'_, CStr>> for Box<CStr> {
    /// Converts a `Cow<'a, CStr>` into a `Box<CStr>`,
    /// by copying the contents if they are borrowed.
    #[inline]
    fn from(cow: Cow<'_, CStr>) -> Box<CStr> {
        match cow {
            Cow::Borrowed(s) => Box::from(s),
            Cow::Owned(s) => Box::from(s),
        }
    }
}

#[stable(feature = "c_string_from_box", since = "1.18.0")]
impl From<Box<CStr>> for CString {
    /// Converts a <code>[Box]<[CStr]></code> into a [`CString`] without copying or allocating.
    #[inline]
    fn from(s: Box<CStr>) -> CString {
        let raw = Box::into_raw(s) as *mut [u8];
        CString { inner: unsafe { Box::from_raw(raw) } }
    }
}

#[stable(feature = "cstring_from_vec_of_nonzerou8", since = "1.43.0")]
impl From<Vec<NonZero<u8>>> for CString {
    /// Converts a <code>[Vec]<[NonZero]<[u8]>></code> into a [`CString`] without
    /// copying nor checking for inner nul bytes.
    #[inline]
    fn from(v: Vec<NonZero<u8>>) -> CString {
        unsafe {
            // Transmute `Vec<NonZero<u8>>` to `Vec<u8>`.
            let v: Vec<u8> = {
                // SAFETY:
                //   - transmuting between `NonZero<u8>` and `u8` is sound;
                //   - `alloc::Layout<NonZero<u8>> == alloc::Layout<u8>`.
                let (ptr, len, cap): (*mut NonZero<u8>, _, _) = Vec::into_raw_parts(v);
                Vec::from_raw_parts(ptr.cast::<u8>(), len, cap)
            };
            // SAFETY: `v` cannot contain nul bytes, given the type-level
            // invariant of `NonZero<u8>`.
            Self::_from_vec_unchecked(v)
        }
    }
}

#[stable(feature = "c_string_from_str", since = "1.85.0")]
impl FromStr for CString {
    type Err = NulError;

    /// Converts a string `s` into a [`CString`].
    ///
    /// This method is equivalent to [`CString::new`].
    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

#[stable(feature = "c_string_from_str", since = "1.85.0")]
impl TryFrom<CString> for String {
    type Error = IntoStringError;

    /// Converts a [`CString`] into a [`String`] if it contains valid UTF-8 data.
    ///
    /// This method is equivalent to [`CString::into_string`].
    #[inline]
    fn try_from(value: CString) -> Result<Self, Self::Error> {
        value.into_string()
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
    /// Converts a [`CString`] into a <code>[Box]<[CStr]></code> without copying or allocating.
    #[inline]
    fn from(s: CString) -> Box<CStr> {
        s.into_boxed_c_str()
    }
}

#[stable(feature = "cow_from_cstr", since = "1.28.0")]
impl<'a> From<CString> for Cow<'a, CStr> {
    /// Converts a [`CString`] into an owned [`Cow`] without copying or allocating.
    #[inline]
    fn from(s: CString) -> Cow<'a, CStr> {
        Cow::Owned(s)
    }
}

#[stable(feature = "cow_from_cstr", since = "1.28.0")]
impl<'a> From<&'a CStr> for Cow<'a, CStr> {
    /// Converts a [`CStr`] into a borrowed [`Cow`] without copying or allocating.
    #[inline]
    fn from(s: &'a CStr) -> Cow<'a, CStr> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_cstr", since = "1.28.0")]
impl<'a> From<&'a CString> for Cow<'a, CStr> {
    /// Converts a `&`[`CString`] into a borrowed [`Cow`] without copying or allocating.
    #[inline]
    fn from(s: &'a CString) -> Cow<'a, CStr> {
        Cow::Borrowed(s.as_c_str())
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<CString> for Arc<CStr> {
    /// Converts a [`CString`] into an <code>[Arc]<[CStr]></code> by moving the [`CString`]
    /// data into a new [`Arc`] buffer.
    #[inline]
    fn from(s: CString) -> Arc<CStr> {
        let arc: Arc<[u8]> = Arc::from(s.into_inner());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const CStr) }
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&CStr> for Arc<CStr> {
    /// Converts a `&CStr` into a `Arc<CStr>`,
    /// by copying the contents into a newly allocated [`Arc`].
    #[inline]
    fn from(s: &CStr) -> Arc<CStr> {
        let arc: Arc<[u8]> = Arc::from(s.to_bytes_with_nul());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const CStr) }
    }
}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl From<&mut CStr> for Arc<CStr> {
    /// Converts a `&mut CStr` into a `Arc<CStr>`,
    /// by copying the contents into a newly allocated [`Arc`].
    #[inline]
    fn from(s: &mut CStr) -> Arc<CStr> {
        Arc::from(&*s)
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<CString> for Rc<CStr> {
    /// Converts a [`CString`] into an <code>[Rc]<[CStr]></code> by moving the [`CString`]
    /// data into a new [`Rc`] buffer.
    #[inline]
    fn from(s: CString) -> Rc<CStr> {
        let rc: Rc<[u8]> = Rc::from(s.into_inner());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const CStr) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&CStr> for Rc<CStr> {
    /// Converts a `&CStr` into a `Rc<CStr>`,
    /// by copying the contents into a newly allocated [`Rc`].
    #[inline]
    fn from(s: &CStr) -> Rc<CStr> {
        let rc: Rc<[u8]> = Rc::from(s.to_bytes_with_nul());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const CStr) }
    }
}

#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl From<&mut CStr> for Rc<CStr> {
    /// Converts a `&mut CStr` into a `Rc<CStr>`,
    /// by copying the contents into a newly allocated [`Rc`].
    #[inline]
    fn from(s: &mut CStr) -> Rc<CStr> {
        Rc::from(&*s)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Rc<CStr> {
    /// Creates an empty CStr inside an Rc
    ///
    /// This may or may not share an allocation with other Rcs on the same thread.
    #[inline]
    fn default() -> Self {
        let rc = Rc::<[u8]>::from(*b"\0");
        // `[u8]` has the same layout as `CStr`, and it is `NUL` terminated.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn nul_position(&self) -> usize {
        self.0
    }

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
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_vec(self) -> Vec<u8> {
        self.1
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for NulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nul byte found in provided data at position: {}", self.0)
    }
}

#[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
impl fmt::Display for FromVecWithNulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.error_kind {
            FromBytesWithNulErrorKind::InteriorNul(pos) => {
                write!(f, "data provided contains an interior nul byte at pos {pos}")
            }
            FromBytesWithNulErrorKind::NotNulTerminated => {
                write!(f, "data provided is not nul terminated")
            }
        }
    }
}

impl IntoStringError {
    /// Consumes this error, returning original [`CString`] which generated the
    /// error.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn into_cstring(self) -> CString {
        self.inner
    }

    /// Access the underlying UTF-8 error that was the cause of this error.
    #[must_use]
    #[stable(feature = "cstring_into", since = "1.7.0")]
    pub fn utf8_error(&self) -> Utf8Error {
        self.error
    }
}

impl IntoStringError {
    fn description(&self) -> &str {
        "C string contained non-utf8 bytes"
    }
}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl fmt::Display for IntoStringError {
    #[allow(deprecated, deprecated_in_future)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.description().fmt(f)
    }
}

#[stable(feature = "cstr_borrow", since = "1.3.0")]
impl ToOwned for CStr {
    type Owned = CString;

    fn to_owned(&self) -> CString {
        CString { inner: self.to_bytes_with_nul().into() }
    }

    fn clone_into(&self, target: &mut CString) {
        let mut b = mem::take(&mut target.inner).into_vec();
        self.to_bytes_with_nul().clone_into(&mut b);
        target.inner = b.into_boxed_slice();
    }
}

#[stable(feature = "cstring_asref", since = "1.7.0")]
impl From<&CStr> for CString {
    /// Converts a <code>&[CStr]</code> into a [`CString`]
    /// by copying the contents into a new allocation.
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
impl AsRef<CStr> for CString {
    #[inline]
    fn as_ref(&self) -> &CStr {
        self
    }
}

impl CStr {
    /// Converts a `CStr` into a <code>[Cow]<[str]></code>.
    ///
    /// If the contents of the `CStr` are valid UTF-8 data, this
    /// function will return a <code>[Cow]::[Borrowed]\(&[str])</code>
    /// with the corresponding <code>&[str]</code> slice. Otherwise, it will
    /// replace any invalid UTF-8 sequences with
    /// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD] and return a
    /// <code>[Cow]::[Owned]\([String])</code> with the result.
    ///
    /// [str]: prim@str "str"
    /// [Borrowed]: Cow::Borrowed
    /// [Owned]: Cow::Owned
    /// [U+FFFD]: core::char::REPLACEMENT_CHARACTER "std::char::REPLACEMENT_CHARACTER"
    ///
    /// # Examples
    ///
    /// Calling `to_string_lossy` on a `CStr` containing valid UTF-8. The leading
    /// `c` on the string literal denotes a `CStr`.
    ///
    /// ```
    /// use std::borrow::Cow;
    ///
    /// assert_eq!(c"Hello World".to_string_lossy(), Cow::Borrowed("Hello World"));
    /// ```
    ///
    /// Calling `to_string_lossy` on a `CStr` containing invalid UTF-8:
    ///
    /// ```
    /// use std::borrow::Cow;
    ///
    /// assert_eq!(
    ///     c"Hello \xF0\x90\x80World".to_string_lossy(),
    ///     Cow::Owned(String::from("Hello �World")) as Cow<'_, str>
    /// );
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "cstr_to_str", since = "1.4.0")]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.to_bytes())
    }

    /// Converts a <code>[Box]<[CStr]></code> into a [`CString`] without copying or allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::{CStr, CString};
    ///
    /// let boxed: Box<CStr> = Box::from(c"foo");
    /// let c_string: CString = c"foo".to_owned();
    ///
    /// assert_eq!(boxed.into_c_string(), c_string);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "into_boxed_c_str", since = "1.20.0")]
    pub fn into_c_string(self: Box<Self>) -> CString {
        CString::from(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::error::Error for NulError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "nul byte found in data"
    }
}

#[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
impl core::error::Error for FromVecWithNulError {}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl core::error::Error for IntoStringError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "C string contained non-utf8 bytes"
    }

    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}
