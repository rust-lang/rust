use crate::fmt;
use crate::panic::Location;

/// A struct providing information about a panic.
///
/// A `PanicInfo` structure is passed to the panic handler defined by `#[panic_handler]`.
///
/// There two `PanicInfo` types:
/// - `core::panic::PanicInfo`, which is used as an argument to a `#[panic_handler]` in `#![no_std]` programs.
/// - [`std::panic::PanicInfo`], which is used as an argument to a panic hook set by [`std::panic::set_hook`].
///
/// This is the first one.
///
/// [`std::panic::set_hook`]: ../../std/panic/fn.set_hook.html
/// [`std::panic::PanicInfo`]: ../../std/panic/struct.PanicInfo.html
#[lang = "panic_info"]
#[stable(feature = "panic_hooks", since = "1.10.0")]
#[derive(Debug)]
pub struct PanicInfo<'a> {
    message: fmt::Arguments<'a>,
    location: &'a Location<'a>,
    can_unwind: bool,
    force_no_backtrace: bool,
}

impl<'a> PanicInfo<'a> {
    #[unstable(
        feature = "panic_internals",
        reason = "internal details of the implementation of the `panic!` and related macros",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn internal_constructor(
        message: fmt::Arguments<'a>,
        location: &'a Location<'a>,
        can_unwind: bool,
        force_no_backtrace: bool,
    ) -> Self {
        PanicInfo { location, message, can_unwind, force_no_backtrace }
    }

    /// If the `panic!` macro from the `core` crate (not from `std`)
    /// was used with a formatting string and some additional arguments,
    /// returns that message ready to be used for example with [`fmt::write`]
    #[must_use]
    #[unstable(feature = "panic_info_message", issue = "66745")]
    pub fn message(&self) -> fmt::Arguments<'_> {
        self.message
    }

    /// Returns information about the location from which the panic originated,
    /// if available.
    ///
    /// This method will currently always return [`Some`], but this may change
    /// in future versions.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(location) = panic_info.location() {
    ///         println!("panic occurred in file '{}' at line {}",
    ///             location.file(),
    ///             location.line(),
    ///         );
    ///     } else {
    ///         println!("panic occurred but can't get location information...");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn location(&self) -> Option<&Location<'_>> {
        // NOTE: If this is changed to sometimes return None,
        // deal with that case in std::panicking::default_hook and core::panicking::panic_fmt.
        Some(&self.location)
    }

    /// Returns the payload associated with the panic.
    ///
    /// On `core::panic::PanicInfo`, this method never returns anything useful.
    /// It only exists because of compatibility with [`std::panic::PanicInfo`],
    /// which used to be the same type.
    ///
    /// See [`std::panic::PanicInfo::payload`].
    ///
    /// [`std::panic::PanicInfo`]: ../../std/panic/struct.PanicInfo.html
    /// [`std::panic::PanicInfo::payload`]: ../../std/panic/struct.PanicInfo.html#method.payload
    #[deprecated(since = "1.74.0", note = "this never returns anything useful")]
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    #[allow(deprecated, deprecated_in_future)]
    pub fn payload(&self) -> &(dyn crate::any::Any + Send) {
        struct NoPayload;
        &NoPayload
    }

    /// Returns whether the panic handler is allowed to unwind the stack from
    /// the point where the panic occurred.
    ///
    /// This is true for most kinds of panics with the exception of panics
    /// caused by trying to unwind out of a `Drop` implementation or a function
    /// whose ABI does not support unwinding.
    ///
    /// It is safe for a panic handler to unwind even when this function returns
    /// false, however this will simply cause the panic handler to be called
    /// again.
    #[must_use]
    #[unstable(feature = "panic_can_unwind", issue = "92988")]
    pub fn can_unwind(&self) -> bool {
        self.can_unwind
    }

    #[unstable(
        feature = "panic_internals",
        reason = "internal details of the implementation of the `panic!` and related macros",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn force_no_backtrace(&self) -> bool {
        self.force_no_backtrace
    }
}

#[stable(feature = "panic_hook_display", since = "1.26.0")]
impl fmt::Display for PanicInfo<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("panicked at ")?;
        self.location.fmt(formatter)?;
        formatter.write_str(":\n")?;
        formatter.write_fmt(self.message)?;
        Ok(())
    }
}
