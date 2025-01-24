use crate::fmt::{self, Display};
use crate::panic::Location;

/// A struct providing information about a panic.
///
/// A `PanicInfo` structure is passed to the panic handler defined by `#[panic_handler]`.
///
/// For the type used by the panic hook mechanism in `std`, see [`std::panic::PanicHookInfo`].
///
/// [`std::panic::PanicHookInfo`]: ../../std/panic/struct.PanicHookInfo.html
#[lang = "panic_info"]
#[stable(feature = "panic_hooks", since = "1.10.0")]
#[derive(Debug)]
pub struct PanicInfo<'a> {
    message: &'a fmt::Arguments<'a>,
    location: &'a Location<'a>,
    can_unwind: bool,
    force_no_backtrace: bool,
}

/// A message that was given to the `panic!()` macro.
///
/// The [`Display`] implementation of this type will format the message with the arguments
/// that were given to the `panic!()` macro.
///
/// See [`PanicInfo::message`].
#[stable(feature = "panic_info_message", since = "1.81.0")]
pub struct PanicMessage<'a> {
    message: &'a fmt::Arguments<'a>,
}

impl<'a> PanicInfo<'a> {
    #[inline]
    pub(crate) fn new(
        message: &'a fmt::Arguments<'a>,
        location: &'a Location<'a>,
        can_unwind: bool,
        force_no_backtrace: bool,
    ) -> Self {
        PanicInfo { location, message, can_unwind, force_no_backtrace }
    }

    /// The message that was given to the `panic!` macro.
    ///
    /// # Example
    ///
    /// The type returned by this method implements `Display`, so it can
    /// be passed directly to [`write!()`] and similar macros.
    ///
    /// [`write!()`]: core::write
    ///
    /// ```ignore (no_std)
    /// #[panic_handler]
    /// fn panic_handler(panic_info: &PanicInfo<'_>) -> ! {
    ///     write!(DEBUG_OUTPUT, "panicked: {}", panic_info.message());
    ///     loop {}
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "panic_info_message", since = "1.81.0")]
    pub fn message(&self) -> PanicMessage<'_> {
        PanicMessage { message: self.message }
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
    /// On this type, `core::panic::PanicInfo`, this method never returns anything useful.
    /// It only exists because of compatibility with [`std::panic::PanicHookInfo`],
    /// which used to be the same type.
    ///
    /// See [`std::panic::PanicHookInfo::payload`].
    ///
    /// [`std::panic::PanicHookInfo`]: ../../std/panic/struct.PanicHookInfo.html
    /// [`std::panic::PanicHookInfo::payload`]: ../../std/panic/struct.PanicHookInfo.html#method.payload
    #[deprecated(since = "1.81.0", note = "this never returns anything useful")]
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
impl Display for PanicInfo<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("panicked at ")?;
        self.location.fmt(formatter)?;
        formatter.write_str(":\n")?;
        formatter.write_fmt(*self.message)?;
        Ok(())
    }
}

impl<'a> PanicMessage<'a> {
    /// Gets the formatted message, if it has no arguments to be formatted at runtime.
    ///
    /// This can be used to avoid allocations in some cases.
    ///
    /// # Guarantees
    ///
    /// For `panic!("just a literal")`, this function is guaranteed to
    /// return `Some("just a literal")`.
    ///
    /// For most cases with placeholders, this function will return `None`.
    ///
    /// See [`fmt::Arguments::as_str`] for details.
    #[stable(feature = "panic_info_message", since = "1.81.0")]
    #[rustc_const_stable(feature = "const_arguments_as_str", since = "1.84.0")]
    #[must_use]
    #[inline]
    pub const fn as_str(&self) -> Option<&'static str> {
        self.message.as_str()
    }
}

#[stable(feature = "panic_info_message", since = "1.81.0")]
impl Display for PanicMessage<'_> {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_fmt(*self.message)
    }
}

#[stable(feature = "panic_info_message", since = "1.81.0")]
impl fmt::Debug for PanicMessage<'_> {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_fmt(*self.message)
    }
}
