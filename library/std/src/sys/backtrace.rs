//! Common code for printing backtraces.
#![forbid(unsafe_op_in_unsafe_fn)]

use crate::backtrace_rs::{self, BacktraceFmt, BytesOrWideString, PrintFmt};
use crate::borrow::Cow;
use crate::io::prelude::*;
use crate::mem::{ManuallyDrop, MaybeUninit};
use crate::path::{self, Path, PathBuf};
use crate::sync::{Mutex, MutexGuard, PoisonError};
use crate::{env, fmt, io};

/// Max number of frames to print.
const MAX_NB_FRAMES: usize = 100;

pub(crate) const FULL_BACKTRACE_DEFAULT: bool = cfg_select! {
    // Fuchsia components default to full backtrace.
    target_os = "fuchsia" => true,
    _ => false,
};

pub(crate) struct BacktraceLock<'a>(#[allow(dead_code)] MutexGuard<'a, ()>);

pub(crate) fn lock<'a>() -> BacktraceLock<'a> {
    static LOCK: Mutex<()> = Mutex::new(());
    BacktraceLock(LOCK.lock().unwrap_or_else(PoisonError::into_inner))
}

impl BacktraceLock<'_> {
    /// Prints the current backtrace.
    pub(crate) fn print(&mut self, w: &mut dyn Write, format: PrintFmt) -> io::Result<()> {
        // There are issues currently linking libbacktrace into tests, and in
        // general during std's own unit tests we're not testing this path. In
        // test mode immediately return here to optimize away any references to the
        // libbacktrace symbols
        if cfg!(test) {
            return Ok(());
        }

        struct DisplayBacktrace {
            format: PrintFmt,
        }
        impl fmt::Display for DisplayBacktrace {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                // SAFETY: the backtrace lock is held
                unsafe { _print_fmt(fmt, self.format) }
            }
        }
        write!(w, "{}", DisplayBacktrace { format })
    }
}

/// # Safety
///
/// This function is not Sync. The caller must hold a mutex lock, or there must be only one thread in the program.
unsafe fn _print_fmt(fmt: &mut fmt::Formatter<'_>, print_fmt: PrintFmt) -> fmt::Result {
    // Always 'fail' to get the cwd when running under Miri -
    // this allows Miri to display backtraces in isolation mode
    let cwd = if !cfg!(miri) { env::current_dir().ok() } else { None };

    let mut print_path = move |fmt: &mut fmt::Formatter<'_>, bows: BytesOrWideString<'_>| {
        output_filename(fmt, bows, print_fmt, cwd.as_ref())
    };
    writeln!(fmt, "stack backtrace:")?;
    let mut bt_fmt = BacktraceFmt::new(fmt, print_fmt, &mut print_path);
    bt_fmt.add_context()?;
    let mut idx = 0;
    let mut res = Ok(());
    let mut omitted_count: usize = 0;
    let mut first_omit = true;
    // If we're using a short backtrace, ignore all frames until we're told to start printing.
    let mut print = print_fmt != PrintFmt::Short;
    set_image_base();
    // SAFETY: we roll our own locking in this town
    unsafe {
        backtrace_rs::trace_unsynchronized(|frame| {
            if print_fmt == PrintFmt::Short && idx > MAX_NB_FRAMES {
                return false;
            }

            if cfg!(feature = "backtrace-trace-only") {
                const HEX_WIDTH: usize = 2 + 2 * size_of::<usize>();
                let frame_ip = frame.ip();
                res = writeln!(bt_fmt.formatter(), "{idx:4}: {frame_ip:HEX_WIDTH$?}");
            } else {
                // `call_with_end_short_backtrace_marker` means we are done hiding symbols
                // for now. Print until we see `call_with_begin_short_backtrace_marker`.
                if print_fmt == PrintFmt::Short {
                    let sym = frame.symbol_address();
                    if sym == call_with_end_short_backtrace_marker as _ {
                        print = true;
                        return true;
                    } else if print && sym == call_with_begin_short_backtrace_marker as _ {
                        print = false;
                        return true;
                    }
                }

                let mut hit = false;
                backtrace_rs::resolve_frame_unsynchronized(frame, |symbol| {
                    hit = true;

                    // Hide `__rust_[begin|end]_short_backtrace` frames from short backtraces.
                    // Unfortunately these generic functions have to be matched by name, as we do
                    // not know their generic parameters.
                    if print_fmt == PrintFmt::Short {
                        if let Some(sym) = symbol.name().and_then(|s| s.as_str()) {
                            if sym.contains("__rust_end_short_backtrace") {
                                print = true;
                                return;
                            }
                            if print && sym.contains("__rust_begin_short_backtrace") {
                                print = false;
                                return;
                            }
                            if !print {
                                omitted_count += 1;
                            }
                        }
                    }

                    if print {
                        if omitted_count > 0 {
                            debug_assert!(print_fmt == PrintFmt::Short);
                            // only print the message between the middle of frames
                            if !first_omit {
                                let _ = writeln!(
                                    bt_fmt.formatter(),
                                    "      [... omitted {} frame{} ...]",
                                    omitted_count,
                                    if omitted_count > 1 { "s" } else { "" }
                                );
                            }
                            first_omit = false;
                            omitted_count = 0;
                        }
                        res = bt_fmt.frame().symbol(frame, symbol);
                    }
                });
                #[cfg(all(target_os = "nto", any(target_env = "nto70", target_env = "nto71")))]
                if libc::__my_thread_exit as *mut libc::c_void == frame.ip() {
                    if !hit && print {
                        use crate::backtrace_rs::SymbolName;
                        res = bt_fmt.frame().print_raw(
                            frame.ip(),
                            Some(SymbolName::new("__my_thread_exit".as_bytes())),
                            None,
                            None,
                        );
                    }
                    return false;
                }
                if !hit && print {
                    res = bt_fmt.frame().print_raw(frame.ip(), None, None, None);
                }
            }

            idx += 1;
            res.is_ok()
        })
    };
    res?;
    bt_fmt.finish()?;
    if print_fmt == PrintFmt::Short {
        writeln!(
            fmt,
            "note: Some details are omitted, \
             run with `RUST_BACKTRACE=full` for a verbose backtrace."
        )?;
    }
    Ok(())
}

macro_rules! short_backtrace_controls {
    ($($adapter:ident => $marker:ident($unique:literal)),* $(,)?) => {$(
        /// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`. Note that
        /// this is only inline(never) when backtraces in std are enabled, otherwise
        /// it's fine to optimize away.
        ///
        /// It is guaranteed that `f` will be called exactly once, and `unsafe` code may
        /// rely on this to be the case.
        #[cfg_attr(feature = "backtrace", inline(never))]
        fn $marker(f: &mut dyn FnMut()) {
            f();

            // (Try to) prevent both Identical Code Folding (which might merge the different
            // versions of this function, giving them the same address) and Tail Call Optimisation
            // (which could remove their frames from the call stack).
            crate::hint::black_box($unique);
        }

        /// Invokes `$marker` with an adaptation of `f`, returning its result.
        /// This is a more ergonomic interface for placing the marker frame on the stack than
        /// the `$marker` function itself. It can be inlined without problem.
        #[doc(hidden)]
        #[unstable(
            feature = "short_backtrace_controls",
            reason = "to control abbreviation of backtraces",
            issue = "none"
        )]
        #[inline(always)]
        pub fn $adapter<F, T>(f: F) -> T
        where
            F: FnOnce() -> T,
        {
            let mut result = MaybeUninit::<T>::uninit();
            let mut f = ManuallyDrop::new(f);

            let mut adapted = || {
                // SAFETY: `adapted` is called exactly once, by `$marker`;
                //         and the `ManuallyDrop` is not otherwise used again.
                let f = unsafe { ManuallyDrop::take(&mut f) };
                result.write(f());
            };

            $marker(&mut adapted);

            // SAFETY: `$marker` guaranteed that it would call `adapted`, which
            //         initialized `result`.
            unsafe { result.assume_init() }
        }
    )*};
}

short_backtrace_controls! {
    __rust_begin_short_backtrace => call_with_begin_short_backtrace_marker(0),
    __rust_end_short_backtrace => call_with_end_short_backtrace_marker(1),
}

/// Prints the filename of the backtrace frame.
///
/// See also `output`.
pub fn output_filename(
    fmt: &mut fmt::Formatter<'_>,
    bows: BytesOrWideString<'_>,
    print_fmt: PrintFmt,
    cwd: Option<&PathBuf>,
) -> fmt::Result {
    let file: Cow<'_, Path> = match bows {
        #[cfg(unix)]
        BytesOrWideString::Bytes(bytes) => {
            use crate::os::unix::prelude::*;
            Path::new(crate::ffi::OsStr::from_bytes(bytes)).into()
        }
        #[cfg(not(unix))]
        BytesOrWideString::Bytes(bytes) => {
            Path::new(crate::str::from_utf8(bytes).unwrap_or("<unknown>")).into()
        }
        #[cfg(windows)]
        BytesOrWideString::Wide(wide) => {
            use crate::os::windows::prelude::*;
            Cow::Owned(crate::ffi::OsString::from_wide(wide).into())
        }
        #[cfg(not(windows))]
        BytesOrWideString::Wide(_wide) => Path::new("<unknown>").into(),
    };
    if print_fmt == PrintFmt::Short && file.is_absolute() {
        if let Some(cwd) = cwd {
            if let Ok(stripped) = file.strip_prefix(&cwd) {
                if let Some(s) = stripped.to_str() {
                    return write!(fmt, ".{}{s}", path::MAIN_SEPARATOR);
                }
            }
        }
    }
    fmt::Display::fmt(&file.display(), fmt)
}

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
pub fn set_image_base() {
    let image_base = crate::os::fortanix_sgx::mem::image_base();
    backtrace_rs::set_image_base(crate::ptr::without_provenance_mut(image_base as _));
}

#[cfg(not(all(target_vendor = "fortanix", target_env = "sgx")))]
pub fn set_image_base() {
    // nothing to do for platforms other than SGX
}
