//! Panic support in the standard library.

#![unstable(feature = "core_panic_info",
            reason = "newly available in libcore",
            issue = "44489")]

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
    location: Location<'a>,
}

impl<'a> PanicInfo<'a> {
    #![unstable(feature = "panic_internals",
                reason = "internal details of the implementation of the `panic!` \
                          and related macros",
                issue = "0")]
    #[doc(hidden)]
    #[inline]
    pub fn internal_constructor(message: Option<&'a fmt::Arguments<'a>>,
                                location: Location<'a>)
                                -> Self {
        struct NoPayload;
        PanicInfo { payload: &NoPayload, location, message }
    }

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
    #[unstable(feature = "panic_info_message", issue = "44489")]
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
#[derive(Debug)]
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub struct Location<'a> {
    file: &'a str,
    line: u32,
    col: u32,
}

impl<'a> Location<'a> {
    #![unstable(feature = "panic_internals",
                reason = "internal details of the implementation of the `panic!` \
                          and related macros",
                issue = "0")]
    #[doc(hidden)]
    pub fn internal_constructor(file: &'a str, line: u32, col: u32) -> Self {
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
#[unstable(feature = "std_internals", issue = "0")]
#[doc(hidden)]
pub unsafe trait BoxMeUp {
    fn box_me_up(&mut self) -> *mut (dyn Any + Send);
    fn get(&mut self) -> &(dyn Any + Send);
}
