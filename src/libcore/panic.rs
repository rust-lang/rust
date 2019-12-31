//! Panic support in the standard library.

#![stable(feature = "core_panic_info", since = "1.41.0")]

use crate::any::Any;
use crate::fmt;

/// A struct providing information about a panic.
///
/// `PanicInfo` structure is passed to a panic hook set by the [`set_hook`]
/// function.
///
/// [`set_hook`]: ../../std/panic/fn.set_hook.html
///
/// # Examples
///
/// ```should_panic
/// use std::panic;
///
/// panic::set_hook(Box::new(|panic_info| {
///     if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
///         println!("panic occurred: {:?}", s);
///     } else {
///         println!("panic occurred");
///     }
/// }));
///
/// panic!("Normal panic");
/// ```
#[lang = "panic_info"]
#[stable(feature = "panic_hooks", since = "1.10.0")]
#[derive(Debug)]
pub struct PanicInfo<'a> {
    payload: &'a (dyn Any + Send),
    message: Option<&'a fmt::Arguments<'a>>,
    location: &'a Location<'a>,
}

impl<'a> PanicInfo<'a> {
    #[unstable(
        feature = "panic_internals",
        reason = "internal details of the implementation of the `panic!` \
                         and related macros",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn internal_constructor(
        message: Option<&'a fmt::Arguments<'a>>,
        location: &'a Location<'a>,
    ) -> Self {
        struct NoPayload;
        PanicInfo { location, message, payload: &NoPayload }
    }

    #[unstable(
        feature = "panic_internals",
        reason = "internal details of the implementation of the `panic!` \
                         and related macros",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn set_payload(&mut self, info: &'a (dyn Any + Send)) {
        self.payload = info;
    }

    /// Returns the payload associated with the panic.
    ///
    /// This will commonly, but not always, be a `&'static str` or [`String`].
    ///
    /// [`String`]: ../../std/string/struct.String.html
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     println!("panic occurred: {:?}", panic_info.payload().downcast_ref::<&str>().unwrap());
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn payload(&self) -> &(dyn Any + Send) {
        self.payload
    }

    /// If the `panic!` macro from the `core` crate (not from `std`)
    /// was used with a formatting string and some additional arguments,
    /// returns that message ready to be used for example with [`fmt::write`]
    ///
    /// [`fmt::write`]: ../fmt/fn.write.html
    #[unstable(feature = "panic_info_message", issue = "66745")]
    pub fn message(&self) -> Option<&fmt::Arguments<'_>> {
        self.message
    }

    /// Returns information about the location from which the panic originated,
    /// if available.
    ///
    /// This method will currently always return [`Some`], but this may change
    /// in future versions.
    ///
    /// [`Some`]: ../../std/option/enum.Option.html#variant.Some
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(location) = panic_info.location() {
    ///         println!("panic occurred in file '{}' at line {}", location.file(),
    ///             location.line());
    ///     } else {
    ///         println!("panic occurred but can't get location information...");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn location(&self) -> Option<&Location<'_>> {
        // NOTE: If this is changed to sometimes return None,
        // deal with that case in std::panicking::default_hook and std::panicking::begin_panic_fmt.
        Some(&self.location)
    }
}

#[stable(feature = "panic_hook_display", since = "1.26.0")]
impl fmt::Display for PanicInfo<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("panicked at ")?;
        if let Some(message) = self.message {
            write!(formatter, "'{}', ", message)?
        } else if let Some(payload) = self.payload.downcast_ref::<&'static str>() {
            write!(formatter, "'{}', ", payload)?
        }
        // NOTE: we cannot use downcast_ref::<String>() here
        // since String is not available in libcore!
        // The payload is a String when `std::panic!` is called with multiple arguments,
        // but in that case the message is also available.

        self.location.fmt(formatter)
    }
}

/// A struct containing information about the location of a panic.
///
/// This structure is created by the [`location`] method of [`PanicInfo`].
///
/// [`location`]: ../../std/panic/struct.PanicInfo.html#method.location
/// [`PanicInfo`]: ../../std/panic/struct.PanicInfo.html
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
#[lang = "panic_location"]
#[derive(Debug)]
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub struct Location<'a> {
    file: &'a str,
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
    /// ```
    /// #![feature(track_caller)]
    /// use core::panic::Location;
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
    /// assert_eq!(fixed_location.line(), 15);
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
    /// assert_eq!(this_location.line(), 29);
    /// assert_eq!(this_location.column(), 21);
    ///
    /// // running the tracked function in a different location produces a different value
    /// let another_location = get_caller_location();
    /// assert_eq!(this_location.file(), another_location.file());
    /// assert_ne!(this_location.line(), another_location.line());
    /// assert_ne!(this_location.column(), another_location.column());
    /// ```
    #[unstable(
        feature = "track_caller",
        reason = "uses #[track_caller] which is not yet stable",
        issue = "47809"
    )]
    #[track_caller]
    pub const fn caller() -> &'static Location<'static> {
        crate::intrinsics::caller_location()
    }
}

impl<'a> Location<'a> {
    #![unstable(
        feature = "panic_internals",
        reason = "internal details of the implementation of the `panic!` \
                          and related macros",
        issue = "none"
    )]
    #[doc(hidden)]
    pub const fn internal_constructor(file: &'a str, line: u32, col: u32) -> Self {
        Location { file, line, col }
    }

    /// Returns the name of the source file from which the panic originated.
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
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn file(&self) -> &str {
        self.file
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
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn line(&self) -> u32 {
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
    #[stable(feature = "panic_col", since = "1.25.0")]
    pub fn column(&self) -> u32 {
        self.col
    }
}

#[stable(feature = "panic_hook_display", since = "1.26.0")]
impl fmt::Display for Location<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}:{}", self.file, self.line, self.col)
    }
}

/// An internal trait used by libstd to pass data from libstd to `panic_unwind`
/// and other panic runtimes. Not intended to be stabilized any time soon, do
/// not use.
#[unstable(feature = "std_internals", issue = "none")]
#[doc(hidden)]
pub unsafe trait BoxMeUp {
    /// Take full ownership of the contents.
    /// The return type is actually `Box<dyn Any + Send>`, but we cannot use `Box` in libcore.
    ///
    /// After this method got called, only some dummy default value is left in `self`.
    /// Calling this method twice, or calling `get` after calling this method, is an error.
    ///
    /// The argument is borrowed because the panic runtime (`__rust_start_panic`) only
    /// gets a borrowed `dyn BoxMeUp`.
    fn take_box(&mut self) -> *mut (dyn Any + Send);

    /// Just borrow the contents.
    fn get(&mut self) -> &(dyn Any + Send);
}
