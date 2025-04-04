//! Support for capturing a stack backtrace of an OS thread
//!
//! This module contains the support necessary to capture a stack backtrace of a
//! running OS thread from the OS thread itself. The `Backtrace` type supports
//! capturing a stack trace via the `Backtrace::capture` and
//! `Backtrace::force_capture` functions.
//!
//! A backtrace is typically quite handy to attach to errors (e.g. types
//! implementing `std::error::Error`) to get a causal chain of where an error
//! was generated.
//!
//! ## Accuracy
//!
//! Backtraces are attempted to be as accurate as possible, but no guarantees
//! are provided about the exact accuracy of a backtrace. Instruction pointers,
//! symbol names, filenames, line numbers, etc, may all be incorrect when
//! reported. Accuracy is attempted on a best-effort basis, however, any bug
//! reports are always welcome to indicate areas of improvement!
//!
//! For most platforms a backtrace with a filename/line number requires that
//! programs be compiled with debug information. Without debug information
//! filenames/line numbers will not be reported.
//!
//! ## Platform support
//!
//! Not all platforms that std compiles for support capturing backtraces. Some
//! platforms simply do nothing when capturing a backtrace. To check whether the
//! platform supports capturing backtraces you can consult the `BacktraceStatus`
//! enum as a result of `Backtrace::status`.
//!
//! Like above with accuracy platform support is done on a best effort basis.
//! Sometimes libraries might not be available at runtime or something may go
//! wrong which would cause a backtrace to not be captured. Please feel free to
//! report issues with platforms where a backtrace cannot be captured though!
//!
//! ## Environment Variables
//!
//! The `Backtrace::capture` function might not actually capture a backtrace by
//! default. Its behavior is governed by two environment variables:
//!
//! * `RUST_LIB_BACKTRACE` - if this is set to `0` then `Backtrace::capture`
//!   will never capture a backtrace. Any other value set will enable
//!   `Backtrace::capture`.
//!
//! * `RUST_BACKTRACE` - if `RUST_LIB_BACKTRACE` is not set, then this variable
//!   is consulted with the same rules of `RUST_LIB_BACKTRACE`.
//!
//! * If neither of the above env vars are set, then `Backtrace::capture` will
//!   be disabled.
//!
//! Capturing a backtrace can be a quite expensive runtime operation, so the
//! environment variables allow either forcibly disabling this runtime
//! performance hit or allow selectively enabling it in some programs.
//!
//! Note that the `Backtrace::force_capture` function can be used to ignore
//! these environment variables. Also note that the state of environment
//! variables is cached once the first backtrace is created, so altering
//! `RUST_LIB_BACKTRACE` or `RUST_BACKTRACE` at runtime might not actually change
//! how backtraces are captured.

#![stable(feature = "backtrace", since = "1.65.0")]

#[cfg(test)]
mod tests;

// NB: A note on resolution of a backtrace:
//
// Backtraces primarily happen in two steps, one is where we actually capture
// the stack backtrace, giving us a list of instruction pointers corresponding
// to stack frames. Next we take these instruction pointers and, one-by-one,
// turn them into a human readable name (like `main`).
//
// The first phase can be somewhat expensive (walking the stack), especially
// on MSVC where debug information is consulted to return inline frames each as
// their own frame. The second phase, however, is almost always extremely
// expensive (on the order of milliseconds sometimes) when it's consulting debug
// information.
//
// We attempt to amortize this cost as much as possible by delaying resolution
// of an address to a human readable name for as long as possible. When
// `Backtrace::create` is called to capture a backtrace it doesn't actually
// perform any symbol resolution, but rather we lazily resolve symbols only just
// before they're needed for printing. This way we can make capturing a
// backtrace and throwing it away much cheaper, but actually printing a
// backtrace is still basically the same cost.
//
// This strategy comes at the cost of some synchronization required inside of a
// `Backtrace`, but that's a relatively small price to pay relative to capturing
// a backtrace or actually symbolizing it.

use crate::backtrace_rs::{self, BytesOrWideString};
use crate::ffi::c_void;
use crate::panic::UnwindSafe;
use crate::sync::LazyLock;
use crate::sync::atomic::AtomicU8;
use crate::sync::atomic::Ordering::Relaxed;
use crate::sys::backtrace::{lock, output_filename, set_image_base};
use crate::{env, fmt};

/// A captured OS thread stack backtrace.
///
/// This type represents a stack backtrace for an OS thread captured at a
/// previous point in time. In some instances the `Backtrace` type may
/// internally be empty due to configuration. For more information see
/// `Backtrace::capture`.
#[stable(feature = "backtrace", since = "1.65.0")]
#[must_use]
pub struct Backtrace {
    inner: Inner,
}

/// The current status of a backtrace, indicating whether it was captured or
/// whether it is empty for some other reason.
#[stable(feature = "backtrace", since = "1.65.0")]
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum BacktraceStatus {
    /// Capturing a backtrace is not supported, likely because it's not
    /// implemented for the current platform.
    #[stable(feature = "backtrace", since = "1.65.0")]
    Unsupported,
    /// Capturing a backtrace has been disabled through either the
    /// `RUST_LIB_BACKTRACE` or `RUST_BACKTRACE` environment variables.
    #[stable(feature = "backtrace", since = "1.65.0")]
    Disabled,
    /// A backtrace has been captured and the `Backtrace` should print
    /// reasonable information when rendered.
    #[stable(feature = "backtrace", since = "1.65.0")]
    Captured,
}

enum Inner {
    Unsupported,
    Disabled,
    Captured(LazyLock<Capture, LazyResolve>),
}

struct Capture {
    actual_start: usize,
    frames: Vec<BacktraceFrame>,
}

fn _assert_send_sync() {
    fn _assert<T: Send + Sync>() {}
    _assert::<Backtrace>();
}

/// A single frame of a backtrace.
#[unstable(feature = "backtrace_frames", issue = "79676")]
pub struct BacktraceFrame {
    frame: RawFrame,
    symbols: Vec<BacktraceSymbol>,
}

#[derive(Debug)]
enum RawFrame {
    Actual(backtrace_rs::Frame),
    #[cfg(test)]
    Fake,
}

struct BacktraceSymbol {
    name: Option<Vec<u8>>,
    filename: Option<BytesOrWide>,
    lineno: Option<u32>,
    colno: Option<u32>,
}

enum BytesOrWide {
    Bytes(Vec<u8>),
    Wide(Vec<u16>),
}

#[stable(feature = "backtrace", since = "1.65.0")]
impl fmt::Debug for Backtrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let capture = match &self.inner {
            Inner::Unsupported => return fmt.write_str("<unsupported>"),
            Inner::Disabled => return fmt.write_str("<disabled>"),
            Inner::Captured(c) => &**c,
        };

        let frames = &capture.frames[capture.actual_start..];

        write!(fmt, "Backtrace ")?;

        let mut dbg = fmt.debug_list();

        for frame in frames {
            if frame.frame.ip().is_null() {
                continue;
            }

            dbg.entries(&frame.symbols);
        }

        dbg.finish()
    }
}

#[unstable(feature = "backtrace_frames", issue = "79676")]
impl fmt::Debug for BacktraceFrame {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = fmt.debug_list();
        dbg.entries(&self.symbols);
        dbg.finish()
    }
}

impl fmt::Debug for BacktraceSymbol {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME: improve formatting: https://github.com/rust-lang/rust/issues/65280
        // FIXME: Also, include column numbers into the debug format as Display already has them.
        // Until there are stable per-frame accessors, the format shouldn't be changed:
        // https://github.com/rust-lang/rust/issues/65280#issuecomment-638966585
        write!(fmt, "{{ ")?;

        if let Some(fn_name) = self.name.as_ref().map(|b| backtrace_rs::SymbolName::new(b)) {
            write!(fmt, "fn: \"{:#}\"", fn_name)?;
        } else {
            write!(fmt, "fn: <unknown>")?;
        }

        if let Some(fname) = self.filename.as_ref() {
            write!(fmt, ", file: \"{:?}\"", fname)?;
        }

        if let Some(line) = self.lineno {
            write!(fmt, ", line: {:?}", line)?;
        }

        write!(fmt, " }}")
    }
}

impl fmt::Debug for BytesOrWide {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        output_filename(
            fmt,
            match self {
                BytesOrWide::Bytes(w) => BytesOrWideString::Bytes(w),
                BytesOrWide::Wide(w) => BytesOrWideString::Wide(w),
            },
            backtrace_rs::PrintFmt::Short,
            crate::env::current_dir().as_ref().ok(),
        )
    }
}

impl Backtrace {
    /// Returns whether backtrace captures are enabled through environment
    /// variables.
    fn enabled() -> bool {
        // Cache the result of reading the environment variables to make
        // backtrace captures speedy, because otherwise reading environment
        // variables every time can be somewhat slow.
        static ENABLED: AtomicU8 = AtomicU8::new(0);
        match ENABLED.load(Relaxed) {
            0 => {}
            1 => return false,
            _ => return true,
        }
        let enabled = match env::var("RUST_LIB_BACKTRACE") {
            Ok(s) => s != "0",
            Err(_) => match env::var("RUST_BACKTRACE") {
                Ok(s) => s != "0",
                Err(_) => false,
            },
        };
        ENABLED.store(enabled as u8 + 1, Relaxed);
        enabled
    }

    /// Captures a stack backtrace of the current thread.
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
    #[stable(feature = "backtrace", since = "1.65.0")]
    #[inline(never)] // want to make sure there's a frame here to remove
    pub fn capture() -> Backtrace {
        if !Backtrace::enabled() {
            return Backtrace { inner: Inner::Disabled };
        }
        Backtrace::create(Backtrace::capture as usize)
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
    #[stable(feature = "backtrace", since = "1.65.0")]
    #[inline(never)] // want to make sure there's a frame here to remove
    pub fn force_capture() -> Backtrace {
        Backtrace::create(Backtrace::force_capture as usize)
    }

    /// Forcibly captures a disabled backtrace, regardless of environment
    /// variable configuration.
    #[stable(feature = "backtrace", since = "1.65.0")]
    #[rustc_const_stable(feature = "backtrace", since = "1.65.0")]
    pub const fn disabled() -> Backtrace {
        Backtrace { inner: Inner::Disabled }
    }

    // Capture a backtrace which start just before the function addressed by
    // `ip`
    fn create(ip: usize) -> Backtrace {
        let _lock = lock();
        let mut frames = Vec::new();
        let mut actual_start = None;
        set_image_base();
        unsafe {
            backtrace_rs::trace_unsynchronized(|frame| {
                frames.push(BacktraceFrame {
                    frame: RawFrame::Actual(frame.clone()),
                    symbols: Vec::new(),
                });
                if frame.symbol_address().addr() == ip && actual_start.is_none() {
                    actual_start = Some(frames.len());
                }
                true
            });
        }

        // If no frames came out assume that this is an unsupported platform
        // since `backtrace` doesn't provide a way of learning this right now,
        // and this should be a good enough approximation.
        let inner = if frames.is_empty() {
            Inner::Unsupported
        } else {
            Inner::Captured(LazyLock::new(lazy_resolve(Capture {
                actual_start: actual_start.unwrap_or(0),
                frames,
            })))
        };

        Backtrace { inner }
    }

    /// Returns the status of this backtrace, indicating whether this backtrace
    /// request was unsupported, disabled, or a stack trace was actually
    /// captured.
    #[stable(feature = "backtrace", since = "1.65.0")]
    #[must_use]
    pub fn status(&self) -> BacktraceStatus {
        match self.inner {
            Inner::Unsupported => BacktraceStatus::Unsupported,
            Inner::Disabled => BacktraceStatus::Disabled,
            Inner::Captured(_) => BacktraceStatus::Captured,
        }
    }
}

impl<'a> Backtrace {
    /// Returns an iterator over the backtrace frames.
    #[must_use]
    #[unstable(feature = "backtrace_frames", issue = "79676")]
    pub fn frames(&'a self) -> &'a [BacktraceFrame] {
        if let Inner::Captured(c) = &self.inner { &c.frames } else { &[] }
    }
}

#[stable(feature = "backtrace", since = "1.65.0")]
impl fmt::Display for Backtrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let capture = match &self.inner {
            Inner::Unsupported => return fmt.write_str("unsupported backtrace"),
            Inner::Disabled => return fmt.write_str("disabled backtrace"),
            Inner::Captured(c) => &**c,
        };

        let full = fmt.alternate();
        let (frames, style) = if full {
            (&capture.frames[..], backtrace_rs::PrintFmt::Full)
        } else {
            (&capture.frames[capture.actual_start..], backtrace_rs::PrintFmt::Short)
        };

        // When printing paths we try to strip the cwd if it exists, otherwise
        // we just print the path as-is. Note that we also only do this for the
        // short format, because if it's full we presumably want to print
        // everything.
        let cwd = crate::env::current_dir();
        let mut print_path = move |fmt: &mut fmt::Formatter<'_>, path: BytesOrWideString<'_>| {
            output_filename(fmt, path, style, cwd.as_ref().ok())
        };

        let mut f = backtrace_rs::BacktraceFmt::new(fmt, style, &mut print_path);
        f.add_context()?;
        for frame in frames {
            if frame.symbols.is_empty() {
                f.frame().print_raw(frame.frame.ip(), None, None, None)?;
            } else {
                for symbol in frame.symbols.iter() {
                    f.frame().print_raw_with_column(
                        frame.frame.ip(),
                        symbol.name.as_ref().map(|b| backtrace_rs::SymbolName::new(b)),
                        symbol.filename.as_ref().map(|b| match b {
                            BytesOrWide::Bytes(w) => BytesOrWideString::Bytes(w),
                            BytesOrWide::Wide(w) => BytesOrWideString::Wide(w),
                        }),
                        symbol.lineno,
                        symbol.colno,
                    )?;
                }
            }
        }
        f.finish()?;
        Ok(())
    }
}

mod helper {
    use super::*;
    pub(super) type LazyResolve = impl (FnOnce() -> Capture) + Send + Sync + UnwindSafe;

    #[define_opaque(LazyResolve)]
    pub(super) fn lazy_resolve(mut capture: Capture) -> LazyResolve {
        move || {
            // Use the global backtrace lock to synchronize this as it's a
            // requirement of the `backtrace` crate, and then actually resolve
            // everything.
            let _lock = lock();
            for frame in capture.frames.iter_mut() {
                let symbols = &mut frame.symbols;
                let frame = match &frame.frame {
                    RawFrame::Actual(frame) => frame,
                    #[cfg(test)]
                    RawFrame::Fake => unimplemented!(),
                };
                unsafe {
                    backtrace_rs::resolve_frame_unsynchronized(frame, |symbol| {
                        symbols.push(BacktraceSymbol {
                            name: symbol.name().map(|m| m.as_bytes().to_vec()),
                            filename: symbol.filename_raw().map(|b| match b {
                                BytesOrWideString::Bytes(b) => BytesOrWide::Bytes(b.to_owned()),
                                BytesOrWideString::Wide(b) => BytesOrWide::Wide(b.to_owned()),
                            }),
                            lineno: symbol.lineno(),
                            colno: symbol.colno(),
                        });
                    });
                }
            }

            capture
        }
    }
}
use helper::*;

impl RawFrame {
    fn ip(&self) -> *mut c_void {
        match self {
            RawFrame::Actual(frame) => frame.ip(),
            #[cfg(test)]
            RawFrame::Fake => crate::ptr::without_provenance_mut(1),
        }
    }
}
