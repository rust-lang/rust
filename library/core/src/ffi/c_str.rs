use crate::cmp::Ordering;
use crate::error::Error;
use crate::ffi::c_char;
use crate::fmt;
use crate::intrinsics;
use crate::ops;
use crate::slice;
use crate::slice::memchr;
use crate::str;

/// Representation of a borrowed C string.
///
/// This type represents a borrowed reference to a nul-terminated
/// array of bytes. It can be constructed safely from a <code>&[[u8]]</code>
/// slice, or unsafely from a raw `*const c_char`. It can then be
/// converted to a Rust <code>&[str]</code> by performing UTF-8 validation, or
/// into an owned [`CString`].
///
/// `&CStr` is to [`CString`] as <code>&[str]</code> is to [`String`]: the former
/// in each pair are borrowed references; the latter are owned
/// strings.
///
/// Note that this structure is **not** `repr(C)` and is not recommended to be
/// placed in the signatures of FFI functions. Instead, safe wrappers of FFI
/// functions may leverage the unsafe [`CStr::from_ptr`] constructor to provide
/// a safe interface to other consumers.
///
/// [`CString`]: ../../std/ffi/struct.CString.html
/// [`String`]: ../../std/string/struct.String.html
///
/// # Examples
///
/// Inspecting a foreign C string:
///
/// ```ignore (extern-declaration)
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// extern "C" { fn my_string() -> *const c_char; }
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
///     extern "C" { fn work_with(data: *const c_char); }
///
///     unsafe { work_with(data.as_ptr()) }
/// }
///
/// let s = CString::new("data data data data").expect("CString::new failed");
/// work(&s);
/// ```
///
/// Converting a foreign C string into a Rust `String`:
///
/// ```ignore (extern-declaration)
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// extern "C" { fn my_string() -> *const c_char; }
///
/// fn my_string_safe() -> String {
///     let cstr = unsafe { CStr::from_ptr(my_string()) };
///     // Get copy-on-write Cow<'_, str>, then guarantee a freshly-owned String allocation
///     String::from_utf8_lossy(cstr.to_bytes()).to_string()
/// }
///
/// println!("string: {}", my_string_safe());
/// ```
///
/// [str]: prim@str "str"
#[derive(Hash)]
#[cfg_attr(not(test), rustc_diagnostic_item = "CStr")]
#[stable(feature = "core_c_str", since = "1.64.0")]
#[rustc_has_incoherent_inherent_impls]
// FIXME:
// `fn from` in `impl From<&CStr> for Box<CStr>` current implementation relies
// on `CStr` being layout-compatible with `[u8]`.
// When attribute privacy is implemented, `CStr` should be annotated as `#[repr(transparent)]`.
// Anyway, `CStr` representation and layout are considered implementation detail, are
// not documented and must not be relied upon.
pub struct CStr {
    // FIXME: this should not be represented with a DST slice but rather with
    //        just a raw `c_char` along with some form of marker to make
    //        this an unsized type. Essentially `sizeof(&CStr)` should be the
    //        same as `sizeof(&c_char)` but `CStr` should be an unsized type.
    inner: [c_char],
}

/// An error indicating that a nul byte was not in the expected position.
///
/// The slice used to create a [`CStr`] must have one and only one nul byte,
/// positioned at the end.
///
/// This error is created by the [`CStr::from_bytes_with_nul`] method.
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use std::ffi::{CStr, FromBytesWithNulError};
///
/// let _: FromBytesWithNulError = CStr::from_bytes_with_nul(b"f\0oo").unwrap_err();
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "core_c_str", since = "1.64.0")]
pub struct FromBytesWithNulError {
    kind: FromBytesWithNulErrorKind,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum FromBytesWithNulErrorKind {
    InteriorNul(usize),
    NotNulTerminated,
}

impl FromBytesWithNulError {
    const fn interior_nul(pos: usize) -> FromBytesWithNulError {
        FromBytesWithNulError { kind: FromBytesWithNulErrorKind::InteriorNul(pos) }
    }
    const fn not_nul_terminated() -> FromBytesWithNulError {
        FromBytesWithNulError { kind: FromBytesWithNulErrorKind::NotNulTerminated }
    }
}

#[stable(feature = "frombyteswithnulerror_impls", since = "1.17.0")]
impl Error for FromBytesWithNulError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match self.kind {
            FromBytesWithNulErrorKind::InteriorNul(..) => {
                "data provided contains an interior nul byte"
            }
            FromBytesWithNulErrorKind::NotNulTerminated => "data provided is not nul terminated",
        }
    }
}

/// An error indicating that no nul byte was present.
///
/// A slice used to create a [`CStr`] must contain a nul byte somewhere
/// within the slice.
///
/// This error is created by the [`CStr::from_bytes_until_nul`] method.
///
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
pub struct FromBytesUntilNulError(());

#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
impl fmt::Display for FromBytesUntilNulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "data provided does not contain a nul")
    }
}

#[stable(feature = "cstr_debug", since = "1.3.0")]
impl fmt::Debug for CStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.to_bytes().escape_ascii())
    }
}

#[stable(feature = "cstr_default", since = "1.10.0")]
impl Default for &CStr {
    #[inline]
    fn default() -> Self {
        const SLICE: &[c_char] = &[0];
        // SAFETY: `SLICE` is indeed pointing to a valid nul-terminated string.
        unsafe { CStr::from_ptr(SLICE.as_ptr()) }
    }
}

#[stable(feature = "frombyteswithnulerror_impls", since = "1.17.0")]
impl fmt::Display for FromBytesWithNulError {
    #[allow(deprecated, deprecated_in_future)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())?;
        if let FromBytesWithNulErrorKind::InteriorNul(pos) = self.kind {
            write!(f, " at byte pos {pos}")?;
        }
        Ok(())
    }
}

impl CStr {
    /// Wraps a raw C string with a safe C string wrapper.
    ///
    /// This function will wrap the provided `ptr` with a `CStr` wrapper, which
    /// allows inspection and interoperation of non-owned C strings. The total
    /// size of the raw C string must be smaller than `isize::MAX` **bytes**
    /// in memory due to calling the `slice::from_raw_parts` function.
    ///
    /// # Safety
    ///
    /// * The memory pointed to by `ptr` must contain a valid nul terminator at the
    ///   end of the string.
    ///
    /// * `ptr` must be [valid] for reads of bytes up to and including the null terminator.
    ///   This means in particular:
    ///
    ///     * The entire memory range of this `CStr` must be contained within a single allocated object!
    ///     * `ptr` must be non-null even for a zero-length cstr.
    ///
    /// * The memory referenced by the returned `CStr` must not be mutated for
    ///   the duration of lifetime `'a`.
    ///
    /// > **Note**: This operation is intended to be a 0-cost cast but it is
    /// > currently implemented with an up-front calculation of the length of
    /// > the string. This is not guaranteed to always be the case.
    ///
    /// # Caveat
    ///
    /// The lifetime for the returned slice is inferred from its usage. To prevent accidental misuse,
    /// it's suggested to tie the lifetime to whichever source lifetime is safe in the context,
    /// such as by providing a helper function taking the lifetime of a host value for the slice,
    /// or by explicit annotation.
    ///
    /// # Examples
    ///
    /// ```ignore (extern-declaration)
    /// use std::ffi::{c_char, CStr};
    ///
    /// extern "C" {
    ///     fn my_string() -> *const c_char;
    /// }
    ///
    /// unsafe {
    ///     let slice = CStr::from_ptr(my_string());
    ///     println!("string returned: {}", slice.to_str().unwrap());
    /// }
    /// ```
    ///
    /// ```
    /// #![feature(const_cstr_methods)]
    ///
    /// use std::ffi::{c_char, CStr};
    ///
    /// const HELLO_PTR: *const c_char = {
    ///     const BYTES: &[u8] = b"Hello, world!\0";
    ///     BYTES.as_ptr().cast()
    /// };
    /// const HELLO: &CStr = unsafe { CStr::from_ptr(HELLO_PTR) };
    /// ```
    ///
    /// [valid]: core::ptr#safety
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_cstr_methods", issue = "101719")]
    pub const unsafe fn from_ptr<'a>(ptr: *const c_char) -> &'a CStr {
        // SAFETY: The caller has provided a pointer that points to a valid C
        // string with a NUL terminator of size less than `isize::MAX`, whose
        // content remain valid and doesn't change for the lifetime of the
        // returned `CStr`.
        //
        // Thus computing the length is fine (a NUL byte exists), the call to
        // from_raw_parts is safe because we know the length is at most `isize::MAX`, meaning
        // the call to `from_bytes_with_nul_unchecked` is correct.
        //
        // The cast from c_char to u8 is ok because a c_char is always one byte.
        unsafe {
            const fn strlen_ct(s: *const c_char) -> usize {
                let mut len = 0;

                // SAFETY: Outer caller has provided a pointer to a valid C string.
                while unsafe { *s.add(len) } != 0 {
                    len += 1;
                }

                len
            }

            fn strlen_rt(s: *const c_char) -> usize {
                extern "C" {
                    /// Provided by libc or compiler_builtins.
                    fn strlen(s: *const c_char) -> usize;
                }

                // SAFETY: Outer caller has provided a pointer to a valid C string.
                unsafe { strlen(s) }
            }

            let len = intrinsics::const_eval_select((ptr,), strlen_ct, strlen_rt);
            Self::from_bytes_with_nul_unchecked(slice::from_raw_parts(ptr.cast(), len + 1))
        }
    }

    /// Creates a C string wrapper from a byte slice.
    ///
    /// This method will create a `CStr` from any byte slice that contains at
    /// least one nul byte. The caller does not need to know or specify where
    /// the nul byte is located.
    ///
    /// If the first byte is a nul character, this method will return an
    /// empty `CStr`. If multiple nul characters are present, the `CStr` will
    /// end at the first one.
    ///
    /// If the slice only has a single nul byte at the end, this method is
    /// equivalent to [`CStr::from_bytes_with_nul`].
    ///
    /// # Examples
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let mut buffer = [0u8; 16];
    /// unsafe {
    ///     // Here we might call an unsafe C function that writes a string
    ///     // into the buffer.
    ///     let buf_ptr = buffer.as_mut_ptr();
    ///     buf_ptr.write_bytes(b'A', 8);
    /// }
    /// // Attempt to extract a C nul-terminated string from the buffer.
    /// let c_str = CStr::from_bytes_until_nul(&buffer[..]).unwrap();
    /// assert_eq!(c_str.to_str().unwrap(), "AAAAAAAA");
    /// ```
    ///
    #[rustc_allow_const_fn_unstable(const_slice_index)]
    #[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
    #[rustc_const_stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
    pub const fn from_bytes_until_nul(bytes: &[u8]) -> Result<&CStr, FromBytesUntilNulError> {
        let nul_pos = memchr::memchr(0, bytes);
        match nul_pos {
            Some(nul_pos) => {
                let subslice = &bytes[..nul_pos + 1];
                // SAFETY: We know there is a nul byte at nul_pos, so this slice
                // (ending at the nul byte) is a well-formed C string.
                Ok(unsafe { CStr::from_bytes_with_nul_unchecked(subslice) })
            }
            None => Err(FromBytesUntilNulError(())),
        }
    }

    /// Creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CStr`
    /// wrapper after ensuring that the byte slice is nul-terminated
    /// and does not contain any interior nul bytes.
    ///
    /// If the nul byte may not be at the end,
    /// [`CStr::from_bytes_until_nul`] can be used instead.
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
    /// let cstr = CStr::from_bytes_with_nul(b"hello");
    /// assert!(cstr.is_err());
    /// ```
    ///
    /// Creating a `CStr` with an interior nul byte is an error:
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let cstr = CStr::from_bytes_with_nul(b"he\0llo\0");
    /// assert!(cstr.is_err());
    /// ```
    #[stable(feature = "cstr_from_bytes", since = "1.10.0")]
    #[rustc_const_unstable(feature = "const_cstr_methods", issue = "101719")]
    pub const fn from_bytes_with_nul(bytes: &[u8]) -> Result<&Self, FromBytesWithNulError> {
        let nul_pos = memchr::memchr(0, bytes);
        match nul_pos {
            Some(nul_pos) if nul_pos + 1 == bytes.len() => {
                // SAFETY: We know there is only one nul byte, at the end
                // of the byte slice.
                Ok(unsafe { Self::from_bytes_with_nul_unchecked(bytes) })
            }
            Some(nul_pos) => Err(FromBytesWithNulError::interior_nul(nul_pos)),
            None => Err(FromBytesWithNulError::not_nul_terminated()),
        }
    }

    /// Unsafely creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CStr` wrapper without
    /// performing any sanity checks.
    ///
    /// # Safety
    /// The provided slice **must** be nul-terminated and not contain any interior
    /// nul bytes.
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
    #[must_use]
    #[stable(feature = "cstr_from_bytes", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_cstr_unchecked", since = "1.59.0")]
    #[rustc_allow_const_fn_unstable(const_eval_select)]
    pub const unsafe fn from_bytes_with_nul_unchecked(bytes: &[u8]) -> &CStr {
        #[inline]
        fn rt_impl(bytes: &[u8]) -> &CStr {
            // Chance at catching some UB at runtime with debug builds.
            debug_assert!(!bytes.is_empty() && bytes[bytes.len() - 1] == 0);

            // SAFETY: Casting to CStr is safe because its internal representation
            // is a [u8] too (safe only inside std).
            // Dereferencing the obtained pointer is safe because it comes from a
            // reference. Making a reference is then safe because its lifetime
            // is bound by the lifetime of the given `bytes`.
            unsafe { &*(bytes as *const [u8] as *const CStr) }
        }

        const fn const_impl(bytes: &[u8]) -> &CStr {
            // Saturating so that an empty slice panics in the assert with a good
            // message, not here due to underflow.
            let mut i = bytes.len().saturating_sub(1);
            assert!(!bytes.is_empty() && bytes[i] == 0, "input was not nul-terminated");

            // Ending null byte exists, skip to the rest.
            while i != 0 {
                i -= 1;
                let byte = bytes[i];
                assert!(byte != 0, "input contained interior nul");
            }

            // SAFETY: See `rt_impl` cast.
            unsafe { &*(bytes as *const [u8] as *const CStr) }
        }

        // SAFETY: The const and runtime versions have identical behavior
        // unless the safety contract of `from_bytes_with_nul_unchecked` is
        // violated, which is UB.
        unsafe { intrinsics::const_eval_select((bytes,), const_impl, rt_impl) }
    }

    /// Returns the inner pointer to this C string.
    ///
    /// The returned pointer will be valid for as long as `self` is, and points
    /// to a contiguous region of memory terminated with a 0 byte to represent
    /// the end of the string.
    ///
    /// The type of the returned pointer is
    /// [`*const c_char`][crate::ffi::c_char], and whether it's
    /// an alias for `*const i8` or `*const u8` is platform-specific.
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
    /// # #![allow(unused_must_use)] #![allow(temporary_cstring_as_ptr)]
    /// use std::ffi::CString;
    ///
    /// // Do not do this:
    /// let ptr = CString::new("Hello").expect("CString::new failed").as_ptr();
    /// unsafe {
    ///     // `ptr` is dangling
    ///     *ptr;
    /// }
    /// ```
    ///
    /// This happens because the pointer returned by `as_ptr` does not carry any
    /// lifetime information and the `CString` is deallocated immediately after
    /// the `CString::new("Hello").expect("CString::new failed").as_ptr()`
    /// expression is evaluated.
    /// To fix the problem, bind the `CString` to a local variable:
    ///
    /// ```no_run
    /// # #![allow(unused_must_use)]
    /// use std::ffi::CString;
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
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_str_as_ptr", since = "1.32.0")]
    pub const fn as_ptr(&self) -> *const c_char {
        self.inner.as_ptr()
    }

    /// Returns `true` if `self.to_bytes()` has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(cstr_is_empty)]
    ///
    /// use std::ffi::CStr;
    /// # use std::ffi::FromBytesWithNulError;
    ///
    /// # fn main() { test().unwrap(); }
    /// # fn test() -> Result<(), FromBytesWithNulError> {
    /// let cstr = CStr::from_bytes_with_nul(b"foo\0")?;
    /// assert!(!cstr.is_empty());
    ///
    /// let empty_cstr = CStr::from_bytes_with_nul(b"\0")?;
    /// assert!(empty_cstr.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[unstable(feature = "cstr_is_empty", issue = "102444")]
    pub const fn is_empty(&self) -> bool {
        // SAFETY: We know there is at least one byte; for empty strings it
        // is the NUL terminator.
        (unsafe { self.inner.get_unchecked(0) }) == &0
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
    /// let cstr = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(cstr.to_bytes(), b"foo");
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_cstr_methods", issue = "101719")]
    pub const fn to_bytes(&self) -> &[u8] {
        let bytes = self.to_bytes_with_nul();
        // SAFETY: to_bytes_with_nul returns slice with length at least 1
        unsafe { bytes.get_unchecked(..bytes.len() - 1) }
    }

    /// Converts this C string to a byte slice containing the trailing 0 byte.
    ///
    /// This function is the equivalent of [`CStr::to_bytes`] except that it
    /// will retain the trailing nul terminator instead of chopping it off.
    ///
    /// > **Note**: This method is currently implemented as a 0-cost cast, but
    /// > it is planned to alter its definition in the future to perform the
    /// > length calculation whenever this method is called.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let cstr = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(cstr.to_bytes_with_nul(), b"foo\0");
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_cstr_methods", issue = "101719")]
    pub const fn to_bytes_with_nul(&self) -> &[u8] {
        // SAFETY: Transmuting a slice of `c_char`s to a slice of `u8`s
        // is safe on all supported targets.
        unsafe { &*(&self.inner as *const [c_char] as *const [u8]) }
    }

    /// Yields a <code>&[str]</code> slice if the `CStr` contains valid UTF-8.
    ///
    /// If the contents of the `CStr` are valid UTF-8 data, this
    /// function will return the corresponding <code>&[str]</code> slice. Otherwise,
    /// it will return an error with details of where UTF-8 validation failed.
    ///
    /// [str]: prim@str "str"
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::CStr;
    ///
    /// let cstr = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    /// assert_eq!(cstr.to_str(), Ok("foo"));
    /// ```
    #[stable(feature = "cstr_to_str", since = "1.4.0")]
    #[rustc_const_unstable(feature = "const_cstr_methods", issue = "101719")]
    pub const fn to_str(&self) -> Result<&str, str::Utf8Error> {
        // N.B., when `CStr` is changed to perform the length check in `.to_bytes()`
        // instead of in `from_ptr()`, it may be worth considering if this should
        // be rewritten to do the UTF-8 check inline with the length calculation
        // instead of doing it afterwards.
        str::from_utf8(self.to_bytes())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for CStr {
    #[inline]
    fn eq(&self, other: &CStr) -> bool {
        self.to_bytes().eq(other.to_bytes())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for CStr {}
#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for CStr {
    #[inline]
    fn partial_cmp(&self, other: &CStr) -> Option<Ordering> {
        self.to_bytes().partial_cmp(&other.to_bytes())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for CStr {
    #[inline]
    fn cmp(&self, other: &CStr) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

#[stable(feature = "cstr_range_from", since = "1.47.0")]
impl ops::Index<ops::RangeFrom<usize>> for CStr {
    type Output = CStr;

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &CStr {
        let bytes = self.to_bytes_with_nul();
        // we need to manually check the starting index to account for the null
        // byte, since otherwise we could get an empty string that doesn't end
        // in a null.
        if index.start < bytes.len() {
            // SAFETY: Non-empty tail of a valid `CStr` is still a valid `CStr`.
            unsafe { CStr::from_bytes_with_nul_unchecked(&bytes[index.start..]) }
        } else {
            panic!(
                "index out of bounds: the len is {} but the index is {}",
                bytes.len(),
                index.start
            );
        }
    }
}

#[stable(feature = "cstring_asref", since = "1.7.0")]
impl AsRef<CStr> for CStr {
    #[inline]
    fn as_ref(&self) -> &CStr {
        self
    }
}
