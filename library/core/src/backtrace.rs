//! hi
#![unstable(feature = "backtrace", issue = "53487")]
use crate::{fmt, ptr};

/// The current status of a backtrace, indicating whether it was captured or
/// whether it is empty for some other reason.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum BacktraceStatus {
    /// Capturing a backtrace is not supported, likely because it's not
    /// implemented for the current platform.
    Unsupported,
    /// Capturing a backtrace has been disabled through either the
    /// `RUST_LIB_BACKTRACE` or `RUST_BACKTRACE` environment variables.
    Disabled,
    /// A backtrace has been captured and the `Backtrace` should print
    /// reasonable information when rendered.
    Captured,
}

// perma(?)-unstable
#[unstable(feature = "backtrace", issue = "53487")]
///
pub trait RawBacktrace: fmt::Debug + fmt::Display + 'static {
    ///
    unsafe fn drop_and_free(self: *mut Self);
}

struct UnsupportedBacktrace;

impl UnsupportedBacktrace {
    #[allow(dead_code)]
    const fn create() -> Backtrace {
        // don't add members to Self
        let _ = Self {};

        Backtrace { inner: ptr::NonNull::<Self>::dangling().as_ptr() }
    }
}

impl RawBacktrace for UnsupportedBacktrace {
    unsafe fn drop_and_free(self: *mut Self) {}
}

impl fmt::Display for UnsupportedBacktrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str("unsupported backtrace")
    }
}

impl fmt::Debug for UnsupportedBacktrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str("<unsupported>")
    }
}
struct DisabledBacktrace;

impl DisabledBacktrace {
    const fn create() -> Backtrace {
        // don't add members to Self
        let _ = Self {};

        Backtrace { inner: ptr::NonNull::<Self>::dangling().as_ptr() }
    }
}

impl RawBacktrace for DisabledBacktrace {
    unsafe fn drop_and_free(self: *mut Self) {}
}

impl fmt::Display for DisabledBacktrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str("disabled backtrace")
    }
}

impl fmt::Debug for DisabledBacktrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str("<disabled>")
    }
}

#[unstable(feature = "backtrace", issue = "53487")]
///
pub struct Backtrace {
    ///
    inner: *mut dyn RawBacktrace,
}

/// Global implementation of backtrace functionality. Called to create
/// `RawBacktrace` trait objects.
#[cfg(not(bootstrap))]
extern "Rust" {
    #[lang = "backtrace_create"]
    fn backtrace_create(ip: usize) -> *mut dyn RawBacktrace;

    #[lang = "backtrace_enabled"]
    fn backtrace_enabled() -> bool;

    #[lang = "backtrace_status"]
    fn backtrace_status(raw: *mut dyn RawBacktrace) -> BacktraceStatus;
}

#[cfg(bootstrap)]
unsafe fn backtrace_create(_ip: usize) -> *mut dyn RawBacktrace {
    UnsupportedBacktrace::create().inner
}

#[cfg(bootstrap)]
unsafe fn backtrace_enabled() -> bool {
    false
}

#[cfg(bootstrap)]
unsafe fn backtrace_status(_raw: *mut dyn RawBacktrace) -> BacktraceStatus {
    BacktraceStatus::Unsupported
}

impl Backtrace {
    fn create(ip: usize) -> Backtrace {
        // SAFETY: trust me
        let inner = unsafe { backtrace_create(ip) };
        Backtrace { inner }
    }

    /// Returns whether backtrace captures are enabled through environment
    /// variables.
    fn enabled() -> bool {
        // SAFETY: trust me
        unsafe { backtrace_enabled() }
    }

    /// Capture a stack backtrace of the current thread.
    ///
    /// This function will capture a stack backtrace of the current OS thread of
    /// execution, returning a `Backtrace` type which can be later used to print
    /// the entire stack trace or render it to a string.
    ///
    /// This function will be a noop if the `RUST_BACKTRACE` or
    /// `RUST_LIB_BACKTRACE` backtrace variables are both not set. If either
    /// environment variable is set and enabled then this function will actually
    /// capture a backtrace. Capturing a backtrace can be both memory intensive
    /// and slow, so these environment variables allow liberally using
    /// `Backtrace::capture` and only incurring a slowdown when the environment
    /// variables are set.
    ///
    /// To forcibly capture a backtrace regardless of environment variables, use
    /// the `Backtrace::force_capture` function.
    #[inline(never)] // want to make sure there's a frame here to remove
    pub fn capture() -> Backtrace {
        if !Backtrace::enabled() {
            return Backtrace::disabled();
        }

        Self::create(Backtrace::capture as usize)
    }

    /// Forcibly captures a full backtrace, regardless of environment variable
    /// configuration.
    ///
    /// This function behaves the same as `capture` except that it ignores the
    /// values of the `RUST_BACKTRACE` and `RUST_LIB_BACKTRACE` environment
    /// variables, always capturing a backtrace.
    ///
    /// Note that capturing a backtrace can be an expensive operation on some
    /// platforms, so this should be used with caution in performance-sensitive
    /// parts of code.
    #[inline(never)] // want to make sure there's a frame here to remove
    pub fn force_capture() -> Backtrace {
        Self::create(Backtrace::force_capture as usize)
    }

    /// Forcibly captures a disabled backtrace, regardless of environment
    /// variable configuration.
    pub const fn disabled() -> Backtrace {
        DisabledBacktrace::create()
    }

    /// Returns the status of this backtrace, indicating whether this backtrace
    /// request was unsupported, disabled, or a stack trace was actually
    /// captured.
    pub fn status(&self) -> BacktraceStatus {
        // SAFETY: trust me
        unsafe { backtrace_status(self.inner) }
    }
}

#[unstable(feature = "backtrace", issue = "53487")]
unsafe impl Send for Backtrace {}

#[unstable(feature = "backtrace", issue = "53487")]
unsafe impl Sync for Backtrace {}

#[unstable(feature = "backtrace", issue = "53487")]
impl Drop for Backtrace {
    fn drop(&mut self) {
        // SAFETY: trust me
        unsafe { RawBacktrace::drop_and_free(self.inner) }
    }
}

#[unstable(feature = "backtrace", issue = "53487")]
impl fmt::Debug for Backtrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: trust me
        let imp: &dyn RawBacktrace = unsafe { &*self.inner };
        fmt::Debug::fmt(imp, fmt)
    }
}

#[unstable(feature = "backtrace", issue = "53487")]
impl fmt::Display for Backtrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: trust me
        let imp: &dyn RawBacktrace = unsafe { &*self.inner };
        fmt::Display::fmt(imp, fmt)
    }
}
