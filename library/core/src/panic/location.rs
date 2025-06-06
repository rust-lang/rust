use crate::ffi::CStr;
use crate::fmt;

/// A struct containing information about the location of a panic.
///
/// This structure is created by [`PanicHookInfo::location()`] and [`PanicInfo::location()`].
///
/// [`PanicInfo::location()`]: crate::panic::PanicInfo::location
/// [`PanicHookInfo::location()`]: ../../std/panic/struct.PanicHookInfo.html#method.location
///
/// # Examples
///
/// ```should_panic
/// use std::panic;
///
/// panic::set_hook(Box::new(|panic_info| {
///     if let Some(location) = panic_info.location() {
///         println!("panic occurred in file '{}' at line {}", location.file(), location.line());
///     } else {
///         println!("panic occurred but can't get location information...");
///     }
/// }));
///
/// panic!("Normal panic");
/// ```
///
/// # Comparisons
///
/// Comparisons for equality and ordering are made in file, line, then column priority.
/// Files are compared as strings, not `Path`, which could be unexpected.
/// See [`Location::file`]'s documentation for more discussion.
#[lang = "panic_location"]
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub struct Location<'a> {
    // Note: this filename will have exactly one nul byte at its end, but otherwise
    // it must never contain interior nul bytes. This is relied on for the conversion
    // to `CStr` below.
    //
    // The prefix of the string without the trailing nul byte will be a regular UTF8 `str`.
    file_bytes_with_nul: &'a [u8],
    line: u32,
    col: u32,
}

impl<'a> Location<'a> {
    /// Returns the source location of the caller of this function. If that function's caller is
    /// annotated then its call location will be returned, and so on up the stack to the first call
    /// within a non-tracked function body.
    ///
    /// # Examples
    ///
    /// ```standalone_crate
    /// use std::panic::Location;
    ///
    /// /// Returns the [`Location`] at which it is called.
    /// #[track_caller]
    /// fn get_caller_location() -> &'static Location<'static> {
    ///     Location::caller()
    /// }
    ///
    /// /// Returns a [`Location`] from within this function's definition.
    /// fn get_just_one_location() -> &'static Location<'static> {
    ///     get_caller_location()
    /// }
    ///
    /// let fixed_location = get_just_one_location();
    /// assert_eq!(fixed_location.file(), file!());
    /// assert_eq!(fixed_location.line(), 14);
    /// assert_eq!(fixed_location.column(), 5);
    ///
    /// // running the same untracked function in a different location gives us the same result
    /// let second_fixed_location = get_just_one_location();
    /// assert_eq!(fixed_location.file(), second_fixed_location.file());
    /// assert_eq!(fixed_location.line(), second_fixed_location.line());
    /// assert_eq!(fixed_location.column(), second_fixed_location.column());
    ///
    /// let this_location = get_caller_location();
    /// assert_eq!(this_location.file(), file!());
    /// assert_eq!(this_location.line(), 28);
    /// assert_eq!(this_location.column(), 21);
    ///
    /// // running the tracked function in a different location produces a different value
    /// let another_location = get_caller_location();
    /// assert_eq!(this_location.file(), another_location.file());
    /// assert_ne!(this_location.line(), another_location.line());
    /// assert_ne!(this_location.column(), another_location.column());
    /// ```
    #[must_use]
    #[stable(feature = "track_caller", since = "1.46.0")]
    #[rustc_const_stable(feature = "const_caller_location", since = "1.79.0")]
    #[track_caller]
    #[inline]
    pub const fn caller() -> &'static Location<'static> {
        crate::intrinsics::caller_location()
    }

    /// Returns the name of the source file from which the panic originated.
    ///
    /// # `&str`, not `&Path`
    ///
    /// The returned name refers to a source path on the compiling system, but it isn't valid to
    /// represent this directly as a `&Path`. The compiled code may run on a different system with
    /// a different `Path` implementation than the system providing the contents and this library
    /// does not currently have a different "host path" type.
    ///
    /// The most surprising behavior occurs when "the same" file is reachable via multiple paths in
    /// the module system (usually using the `#[path = "..."]` attribute or similar), which can
    /// cause what appears to be identical code to return differing values from this function.
    ///
    /// # Cross-compilation
    ///
    /// This value is not suitable for passing to `Path::new` or similar constructors when the host
    /// platform and target platform differ.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(location) = panic_info.location() {
    ///         println!("panic occurred in file '{}'", location.file());
    ///     } else {
    ///         println!("panic occurred but can't get location information...");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_location_fields", since = "1.79.0")]
    pub const fn file(&self) -> &str {
        let str_len = self.file_bytes_with_nul.len() - 1;
        // SAFETY: `file_bytes_with_nul` without the trailing nul byte is guaranteed to be
        // valid UTF8.
        unsafe { crate::str::from_raw_parts(self.file_bytes_with_nul.as_ptr(), str_len) }
    }

    /// Returns the name of the source file as a nul-terminated `CStr`.
    ///
    /// This is useful for interop with APIs that expect C/C++ `__FILE__` or
    /// `std::source_location::file_name`, both of which return a nul-terminated `const char*`.
    #[must_use]
    #[unstable(feature = "file_with_nul", issue = "141727")]
    #[inline]
    pub const fn file_with_nul(&self) -> &CStr {
        // SAFETY: `file_bytes_with_nul` is guaranteed to have a trailing nul byte and no
        // interior nul bytes.
        unsafe { CStr::from_bytes_with_nul_unchecked(self.file_bytes_with_nul) }
    }

    /// Returns the line number from which the panic originated.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(location) = panic_info.location() {
    ///         println!("panic occurred at line {}", location.line());
    ///     } else {
    ///         println!("panic occurred but can't get location information...");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_location_fields", since = "1.79.0")]
    #[inline]
    pub const fn line(&self) -> u32 {
        self.line
    }

    /// Returns the column from which the panic originated.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(location) = panic_info.location() {
    ///         println!("panic occurred at column {}", location.column());
    ///     } else {
    ///         println!("panic occurred but can't get location information...");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[stable(feature = "panic_col", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_location_fields", since = "1.79.0")]
    #[inline]
    pub const fn column(&self) -> u32 {
        self.col
    }
}

#[stable(feature = "panic_hook_display", since = "1.26.0")]
impl fmt::Display for Location<'_> {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}:{}", self.file(), self.line, self.col)
    }
}
