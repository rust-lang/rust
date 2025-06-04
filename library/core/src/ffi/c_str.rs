//! [`CStr`] and its related types.

use crate::cmp::Ordering;
use crate::error::Error;
use crate::ffi::c_char;
use crate::intrinsics::const_eval_select;
use crate::iter::FusedIterator;
use crate::marker::PhantomData;
use crate::ptr::NonNull;
use crate::slice::memchr;
use crate::{fmt, ops, slice, str};

// FIXME: because this is doc(inline)d, we *have* to use intra-doc links because the actual link
//   depends on where the item is being documented. however, since this is libcore, we can't
//   actually reference libstd or liballoc in intra-doc links. so, the best we can do is remove the
//   links to `CString` and `String` for now until a solution is developed

/// Representation of a borrowed C string.
///
/// This type represents a borrowed reference to a nul-terminated
/// array of bytes. It can be constructed safely from a <code>&[[u8]]</code>
/// slice, or unsafely from a raw `*const c_char`. It can be expressed as a
/// literal in the form `c"Hello world"`.
///
/// The `CStr` can then be converted to a Rust <code>&[str]</code> by performing
/// UTF-8 validation, or into an owned `CString`.
///
/// `&CStr` is to `CString` as <code>&[str]</code> is to `String`: the former
/// in each pair are borrowed references; the latter are owned
/// strings.
///
/// Note that this structure does **not** have a guaranteed layout (the `repr(transparent)`
/// notwithstanding) and should not be placed in the signatures of FFI functions.
/// Instead, safe wrappers of FFI functions may leverage [`CStr::as_ptr`] and the unsafe
/// [`CStr::from_ptr`] constructor to provide a safe interface to other consumers.
///
/// # Examples
///
/// Inspecting a foreign C string:
///
/// ```
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// # /* Extern functions are awkward in doc comments - fake it instead
/// extern "C" { fn my_string() -> *const c_char; }
/// # */ unsafe extern "C" fn my_string() -> *const c_char { c"hello".as_ptr() }
///
/// unsafe {
///     let slice = CStr::from_ptr(my_string());
///     println!("string buffer size without nul terminator: {}", slice.to_bytes().len());
/// }
/// ```
///
/// Passing a Rust-originating C string:
///
/// ```
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// fn work(data: &CStr) {
///     unsafe extern "C" fn work_with(s: *const c_char) {}
///     unsafe { work_with(data.as_ptr()) }
/// }
///
/// let s = c"Hello world!";
/// work(&s);
/// ```
///
/// Converting a foreign C string into a Rust `String`:
///
/// ```
/// use std::ffi::CStr;
/// use std::os::raw::c_char;
///
/// # /* Extern functions are awkward in doc comments - fake it instead
/// extern "C" { fn my_string() -> *const c_char; }
/// # */ unsafe extern "C" fn my_string() -> *const c_char { c"hello".as_ptr() }
///
/// fn my_string_safe() -> String {
///     let cstr = unsafe { CStr::from_ptr(my_string()) };
///     // Get a copy-on-write Cow<'_, str>, then extract the
///     // allocated String (or allocate a fresh one if needed).
///     cstr.to_string_lossy().into_owned()
/// }
///
/// println!("string: {}", my_string_safe());
/// ```
///
/// [str]: prim@str "str"
#[derive(PartialEq, Eq, Hash)]
#[stable(feature = "core_c_str", since = "1.64.0")]
#[rustc_diagnostic_item = "cstr_type"]
#[rustc_has_incoherent_inherent_impls]
#[lang = "CStr"]
// `fn from` in `impl From<&CStr> for Box<CStr>` current implementation relies
// on `CStr` being layout-compatible with `[u8]`.
// However, `CStr` layout is considered an implementation detail and must not be relied upon. We
// want `repr(transparent)` but we don't want it to show up in rustdoc, so we hide it under
// `cfg(doc)`. This is an ad-hoc implementation of attribute privacy.
#[repr(transparent)]
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[stable(feature = "core_c_str", since = "1.64.0")]
pub enum FromBytesWithNulError {
    /// Data provided contains an interior nul byte at byte `position`.
    InteriorNul {
        /// The position of the interior nul byte.
        position: usize,
    },
    /// Data provided is not nul terminated.
    NotNulTerminated,
}

#[stable(feature = "frombyteswithnulerror_impls", since = "1.17.0")]
impl Error for FromBytesWithNulError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match self {
            Self::InteriorNul { .. } => "data provided contains an interior nul byte",
            Self::NotNulTerminated => "data provided is not nul terminated",
        }
    }
}

/// An error indicating that no nul byte was present.
///
/// A slice used to create a [`CStr`] must contain a nul byte somewhere
/// within the slice.
///
/// This error is created by the [`CStr::from_bytes_until_nul`] method.
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
        if let Self::InteriorNul { position } = self {
            write!(f, " at byte pos {position}")?;
        }
        Ok(())
    }
}

impl CStr {
    /// Wraps a raw C string with a safe C string wrapper.
    ///
    /// This function will wrap the provided `ptr` with a `CStr` wrapper, which
    /// allows inspection and interoperation of non-owned C strings. The total
    /// size of the terminated buffer must be smaller than [`isize::MAX`] **bytes**
    /// in memory (a restriction from [`slice::from_raw_parts`]).
    ///
    /// # Safety
    ///
    /// * The memory pointed to by `ptr` must contain a valid nul terminator at the
    ///   end of the string.
    ///
    /// * `ptr` must be [valid] for reads of bytes up to and including the nul terminator.
    ///   This means in particular:
    ///
    ///     * The entire memory range of this `CStr` must be contained within a single allocation!
    ///     * `ptr` must be non-null even for a zero-length cstr.
    ///
    /// * The memory referenced by the returned `CStr` must not be mutated for
    ///   the duration of lifetime `'a`.
    ///
    /// * The nul terminator must be within `isize::MAX` from `ptr`
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
    /// ```
    /// use std::ffi::{c_char, CStr};
    ///
    /// fn my_string() -> *const c_char {
    ///     c"hello".as_ptr()
    /// }
    ///
    /// unsafe {
    ///     let slice = CStr::from_ptr(my_string());
    ///     assert_eq!(slice.to_str().unwrap(), "hello");
    /// }
    /// ```
    ///
    /// ```
    /// use std::ffi::{c_char, CStr};
    ///
    /// const HELLO_PTR: *const c_char = {
    ///     const BYTES: &[u8] = b"Hello, world!\0";
    ///     BYTES.as_ptr().cast()
    /// };
    /// const HELLO: &CStr = unsafe { CStr::from_ptr(HELLO_PTR) };
    ///
    /// assert_eq!(c"Hello, world!", HELLO);
    /// ```
    ///
    /// [valid]: core::ptr#safety
    #[inline] // inline is necessary for codegen to see strlen.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_cstr_from_ptr", since = "1.81.0")]
    pub const unsafe fn from_ptr<'a>(ptr: *const c_char) -> &'a CStr {
        // SAFETY: The caller has provided a pointer that points to a valid C
        // string with a NUL terminator less than `isize::MAX` from `ptr`.
        let len = unsafe { strlen(ptr) };

        // SAFETY: The caller has provided a valid pointer with length less than
        // `isize::MAX`, so `from_raw_parts` is safe. The content remains valid
        // and doesn't change for the lifetime of the returned `CStr`. This
        // means the call to `from_bytes_with_nul_unchecked` is correct.
        //
        // The cast from c_char to u8 is ok because a c_char is always one byte.
        unsafe { Self::from_bytes_with_nul_unchecked(slice::from_raw_parts(ptr.cast(), len + 1)) }
    }

    /// Creates a C string wrapper from a byte slice with any number of nuls.
    ///
    /// This method will create a `CStr` from any byte slice that contains at
    /// least one nul byte. Unlike with [`CStr::from_bytes_with_nul`], the caller
    /// does not need to know where the nul byte is located.
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
    #[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
    #[rustc_const_stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
    pub const fn from_bytes_until_nul(bytes: &[u8]) -> Result<&CStr, FromBytesUntilNulError> {
        let nul_pos = memchr::memchr(0, bytes);
        match nul_pos {
            Some(nul_pos) => {
                // FIXME(const-hack) replace with range index
                // SAFETY: nul_pos + 1 <= bytes.len()
                let subslice = unsafe { crate::slice::from_raw_parts(bytes.as_ptr(), nul_pos + 1) };
                // SAFETY: We know there is a nul byte at nul_pos, so this slice
                // (ending at the nul byte) is a well-formed C string.
                Ok(unsafe { CStr::from_bytes_with_nul_unchecked(subslice) })
            }
            None => Err(FromBytesUntilNulError(())),
        }
    }

    /// Creates a C string wrapper from a byte slice with exactly one nul
    /// terminator.
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
    /// assert_eq!(cstr, Ok(c"hello"));
    /// ```
    ///
    /// Creating a `CStr` without a trailing nul terminator is an error:
    ///
    /// ```
    /// use std::ffi::{CStr, FromBytesWithNulError};
    ///
    /// let cstr = CStr::from_bytes_with_nul(b"hello");
    /// assert_eq!(cstr, Err(FromBytesWithNulError::NotNulTerminated));
    /// ```
    ///
    /// Creating a `CStr` with an interior nul byte is an error:
    ///
    /// ```
    /// use std::ffi::{CStr, FromBytesWithNulError};
    ///
    /// let cstr = CStr::from_bytes_with_nul(b"he\0llo\0");
    /// assert_eq!(cstr, Err(FromBytesWithNulError::InteriorNul { position: 2 }));
    /// ```
    #[stable(feature = "cstr_from_bytes", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_cstr_methods", since = "1.72.0")]
    pub const fn from_bytes_with_nul(bytes: &[u8]) -> Result<&Self, FromBytesWithNulError> {
        let nul_pos = memchr::memchr(0, bytes);
        match nul_pos {
            Some(nul_pos) if nul_pos + 1 == bytes.len() => {
                // SAFETY: We know there is only one nul byte, at the end
                // of the byte slice.
                Ok(unsafe { Self::from_bytes_with_nul_unchecked(bytes) })
            }
            Some(position) => Err(FromBytesWithNulError::InteriorNul { position }),
            None => Err(FromBytesWithNulError::NotNulTerminated),
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
    /// use std::ffi::CStr;
    ///
    /// let bytes = b"Hello world!\0";
    ///
    /// let cstr = unsafe { CStr::from_bytes_with_nul_unchecked(bytes) };
    /// assert_eq!(cstr.to_bytes_with_nul(), bytes);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "cstr_from_bytes", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_cstr_unchecked", since = "1.59.0")]
    #[rustc_allow_const_fn_unstable(const_eval_select)]
    pub const unsafe fn from_bytes_with_nul_unchecked(bytes: &[u8]) -> &CStr {
        const_eval_select!(
            @capture { bytes: &[u8] } -> &CStr:
            if const {
                // Saturating so that an empty slice panics in the assert with a good
                // message, not here due to underflow.
                let mut i = bytes.len().saturating_sub(1);
                assert!(!bytes.is_empty() && bytes[i] == 0, "input was not nul-terminated");

                // Ending nul byte exists, skip to the rest.
                while i != 0 {
                    i -= 1;
                    let byte = bytes[i];
                    assert!(byte != 0, "input contained interior nul");
                }

                // SAFETY: See runtime cast comment below.
                unsafe { &*(bytes as *const [u8] as *const CStr) }
            } else {
                // Chance at catching some UB at runtime with debug builds.
                debug_assert!(!bytes.is_empty() && bytes[bytes.len() - 1] == 0);

                // SAFETY: Casting to CStr is safe because its internal representation
                // is a [u8] too (safe only inside std).
                // Dereferencing the obtained pointer is safe because it comes from a
                // reference. Making a reference is then safe because its lifetime
                // is bound by the lifetime of the given `bytes`.
                unsafe { &*(bytes as *const [u8] as *const CStr) }
            }
        )
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
    /// # #![expect(dangling_pointers_from_temporaries)]
    /// use std::ffi::{CStr, CString};
    ///
    /// // 💀 The meaning of this entire program is undefined,
    /// // 💀 and nothing about its behavior is guaranteed,
    /// // 💀 not even that its behavior resembles the code as written,
    /// // 💀 just because it contains a single instance of undefined behavior!
    ///
    /// // 🚨 creates a dangling pointer to a temporary `CString`
    /// // 🚨 that is deallocated at the end of the statement
    /// let ptr = CString::new("Hi!".to_uppercase()).unwrap().as_ptr();
    ///
    /// // without undefined behavior, you would expect that `ptr` equals:
    /// dbg!(CStr::from_bytes_with_nul(b"HI!\0").unwrap());
    ///
    /// // 🙏 Possibly the program behaved as expected so far,
    /// // 🙏 and this just shows `ptr` is now garbage..., but
    /// // 💀 this violates `CStr::from_ptr`'s safety contract
    /// // 💀 leading to a dereference of a dangling pointer,
    /// // 💀 which is immediate undefined behavior.
    /// // 💀 *BOOM*, you're dead, you're entire program has no meaning.
    /// dbg!(unsafe { CStr::from_ptr(ptr) });
    /// ```
    ///
    /// This happens because, the pointer returned by `as_ptr` does not carry any
    /// lifetime information, and the `CString` is deallocated immediately after
    /// the expression that it is part of has been evaluated.
    /// To fix the problem, bind the `CString` to a local variable:
    ///
    /// ```
    /// use std::ffi::{CStr, CString};
    ///
    /// let c_str = CString::new("Hi!".to_uppercase()).unwrap();
    /// let ptr = c_str.as_ptr();
    ///
    /// assert_eq!(unsafe { CStr::from_ptr(ptr) }, c"HI!");
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_str_as_ptr", since = "1.32.0")]
    #[rustc_as_ptr]
    #[rustc_never_returns_null_ptr]
    pub const fn as_ptr(&self) -> *const c_char {
        self.inner.as_ptr()
    }

    /// We could eventually expose this publicly, if we wanted.
    #[inline]
    #[must_use]
    const fn as_non_null_ptr(&self) -> NonNull<c_char> {
        // FIXME(const_trait_impl) replace with `NonNull::from`
        // SAFETY: a reference is never null
        unsafe { NonNull::new_unchecked(&self.inner as *const [c_char] as *mut [c_char]) }
            .as_non_null_ptr()
    }

    /// Returns the length of `self`. Like C's `strlen`, this does not include the nul terminator.
    ///
    /// > **Note**: This method is currently implemented as a constant-time
    /// > cast, but it is planned to alter its definition in the future to
    /// > perform the length calculation whenever this method is called.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(c"foo".count_bytes(), 3);
    /// assert_eq!(c"".count_bytes(), 0);
    /// ```
    #[inline]
    #[must_use]
    #[doc(alias("len", "strlen"))]
    #[stable(feature = "cstr_count_bytes", since = "1.79.0")]
    #[rustc_const_stable(feature = "const_cstr_from_ptr", since = "1.81.0")]
    pub const fn count_bytes(&self) -> usize {
        self.inner.len() - 1
    }

    /// Returns `true` if `self.to_bytes()` has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(!c"foo".is_empty());
    /// assert!(c"".is_empty());
    /// ```
    #[inline]
    #[stable(feature = "cstr_is_empty", since = "1.71.0")]
    #[rustc_const_stable(feature = "cstr_is_empty", since = "1.71.0")]
    pub const fn is_empty(&self) -> bool {
        // SAFETY: We know there is at least one byte; for empty strings it
        // is the NUL terminator.
        // FIXME(const-hack): use get_unchecked
        unsafe { *self.inner.as_ptr() == 0 }
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
    /// assert_eq!(c"foo".to_bytes(), b"foo");
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_cstr_methods", since = "1.72.0")]
    pub const fn to_bytes(&self) -> &[u8] {
        let bytes = self.to_bytes_with_nul();
        // FIXME(const-hack) replace with range index
        // SAFETY: to_bytes_with_nul returns slice with length at least 1
        unsafe { slice::from_raw_parts(bytes.as_ptr(), bytes.len() - 1) }
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
    /// assert_eq!(c"foo".to_bytes_with_nul(), b"foo\0");
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_cstr_methods", since = "1.72.0")]
    pub const fn to_bytes_with_nul(&self) -> &[u8] {
        // SAFETY: Transmuting a slice of `c_char`s to a slice of `u8`s
        // is safe on all supported targets.
        unsafe { &*((&raw const self.inner) as *const [u8]) }
    }

    /// Iterates over the bytes in this C string.
    ///
    /// The returned iterator will **not** contain the trailing nul terminator
    /// that this C string has.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(cstr_bytes)]
    ///
    /// assert!(c"foo".bytes().eq(*b"foo"));
    /// ```
    #[inline]
    #[unstable(feature = "cstr_bytes", issue = "112115")]
    pub fn bytes(&self) -> Bytes<'_> {
        Bytes::new(self)
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
    /// assert_eq!(c"foo".to_str(), Ok("foo"));
    /// ```
    #[stable(feature = "cstr_to_str", since = "1.4.0")]
    #[rustc_const_stable(feature = "const_cstr_methods", since = "1.72.0")]
    pub const fn to_str(&self) -> Result<&str, str::Utf8Error> {
        // N.B., when `CStr` is changed to perform the length check in `.to_bytes()`
        // instead of in `from_ptr()`, it may be worth considering if this should
        // be rewritten to do the UTF-8 check inline with the length calculation
        // instead of doing it afterwards.
        str::from_utf8(self.to_bytes())
    }

    /// Returns an object that implements [`Display`] for safely printing a [`CStr`] that may
    /// contain non-Unicode data.
    ///
    /// Behaves as if `self` were first lossily converted to a `str`, with invalid UTF-8 presented
    /// as the Unicode replacement character: �.
    ///
    /// [`Display`]: fmt::Display
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(cstr_display)]
    ///
    /// let cstr = c"Hello, world!";
    /// println!("{}", cstr.display());
    /// ```
    #[unstable(feature = "cstr_display", issue = "139984")]
    #[must_use = "this does not display the `CStr`; \
                  it returns an object that can be displayed"]
    #[inline]
    pub fn display(&self) -> impl fmt::Display {
        crate::bstr::ByteStr::from_bytes(self.to_bytes())
    }
}

// `.to_bytes()` representations are compared instead of the inner `[c_char]`s,
// because `c_char` is `i8` (not `u8`) on some platforms.
// That is why this is implemented manually and not derived.
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

/// Calculate the length of a nul-terminated string. Defers to C's `strlen` when possible.
///
/// # Safety
///
/// The pointer must point to a valid buffer that contains a NUL terminator. The NUL must be
/// located within `isize::MAX` from `ptr`.
#[inline]
#[unstable(feature = "cstr_internals", issue = "none")]
#[rustc_allow_const_fn_unstable(const_eval_select)]
const unsafe fn strlen(ptr: *const c_char) -> usize {
    const_eval_select!(
        @capture { s: *const c_char = ptr } -> usize:
        if const {
            let mut len = 0;

            // SAFETY: Outer caller has provided a pointer to a valid C string.
            while unsafe { *s.add(len) } != 0 {
                len += 1;
            }

            len
        } else {
            unsafe extern "C" {
                /// Provided by libc or compiler_builtins.
                fn strlen(s: *const c_char) -> usize;
            }

            // SAFETY: Outer caller has provided a pointer to a valid C string.
            unsafe { strlen(s) }
        }
    )
}

/// An iterator over the bytes of a [`CStr`], without the nul terminator.
///
/// This struct is created by the [`bytes`] method on [`CStr`].
/// See its documentation for more.
///
/// [`bytes`]: CStr::bytes
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "cstr_bytes", issue = "112115")]
#[derive(Clone, Debug)]
pub struct Bytes<'a> {
    // since we know the string is nul-terminated, we only need one pointer
    ptr: NonNull<u8>,
    phantom: PhantomData<&'a [c_char]>,
}

#[unstable(feature = "cstr_bytes", issue = "112115")]
unsafe impl Send for Bytes<'_> {}

#[unstable(feature = "cstr_bytes", issue = "112115")]
unsafe impl Sync for Bytes<'_> {}

impl<'a> Bytes<'a> {
    #[inline]
    fn new(s: &'a CStr) -> Self {
        Self { ptr: s.as_non_null_ptr().cast(), phantom: PhantomData }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        // SAFETY: We uphold that the pointer is always valid to dereference
        // by starting with a valid C string and then never incrementing beyond
        // the nul terminator.
        unsafe { self.ptr.read() == 0 }
    }
}

#[unstable(feature = "cstr_bytes", issue = "112115")]
impl Iterator for Bytes<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        // SAFETY: We only choose a pointer from a valid C string, which must
        // be non-null and contain at least one value. Since we always stop at
        // the nul terminator, which is guaranteed to exist, we can assume that
        // the pointer is non-null and valid. This lets us safely dereference
        // it and assume that adding 1 will create a new, non-null, valid
        // pointer.
        unsafe {
            let ret = self.ptr.read();
            if ret == 0 {
                None
            } else {
                self.ptr = self.ptr.add(1);
                Some(ret)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_empty() { (0, Some(0)) } else { (1, None) }
    }

    #[inline]
    fn count(self) -> usize {
        // SAFETY: We always hold a valid pointer to a C string
        unsafe { strlen(self.ptr.as_ptr().cast()) }
    }
}

#[unstable(feature = "cstr_bytes", issue = "112115")]
impl FusedIterator for Bytes<'_> {}
