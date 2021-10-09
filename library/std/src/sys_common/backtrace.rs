use crate::backtrace_rs::{self, BacktraceFmt, BytesOrWideString, PrintFmt};
use crate::borrow::Cow;
/// Common code for printing the backtrace in the same way across the different
/// supported platforms.
use crate::env;
use crate::fmt;
use crate::io;
use crate::io::prelude::*;
use crate::path::{self, Path, PathBuf};
use crate::sync::atomic::{self, Ordering};
use crate::sys_common::mutex::StaticMutex;

/// Max number of frames to print.
const MAX_NB_FRAMES: usize = 100;

// SAFETY: Don't attempt to lock this reentrantly.
pub unsafe fn lock() -> impl Drop {
    static LOCK: StaticMutex = StaticMutex::new();
    LOCK.lock()
}

/// Prints the current backtrace.
pub fn print(w: &mut dyn Write, format: PrintFmt) -> io::Result<()> {
    // There are issues currently linking libbacktrace into tests, and in
    // general during libstd's own unit tests we're not testing this path. In
    // test mode immediately return here to optimize away any references to the
    // libbacktrace symbols
    if cfg!(test) {
        return Ok(());
    }

    // Use a lock to prevent mixed output in multithreading context.
    // Some platforms also requires it, like `SymFromAddr` on Windows.
    unsafe {
        let _lock = lock();
        _print(w, format)
    }
}

unsafe fn _print(w: &mut dyn Write, format: PrintFmt) -> io::Result<()> {
    struct DisplayBacktrace {
        format: PrintFmt,
    }
    impl fmt::Display for DisplayBacktrace {
        fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            unsafe { _print_fmt(fmt, self.format) }
        }
    }
    write!(w, "{}", DisplayBacktrace { format })
}

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
    // Start immediately if we're not using a short backtrace.
    let mut start = print_fmt != PrintFmt::Short;
    backtrace_rs::trace_unsynchronized(|frame| {
        if print_fmt == PrintFmt::Short && idx > MAX_NB_FRAMES {
            return false;
        }

        let mut hit = false;
        let mut stop = false;
        backtrace_rs::resolve_frame_unsynchronized(frame, |symbol| {
            hit = true;
            if print_fmt == PrintFmt::Short {
                if let Some(sym) = symbol.name().and_then(|s| s.as_str()) {
                    if start && sym.contains("__rust_begin_short_backtrace") {
                        stop = true;
                        return;
                    }
                    if sym.contains("__rust_end_short_backtrace") {
                        start = true;
                        return;
                    }
                }
            }

            if start {
                res = bt_fmt.frame().symbol(frame, symbol);
            }
        });
        if stop {
            return false;
        }
        if !hit && start {
            res = bt_fmt.frame().print_raw(frame.ip(), None, None, None);
        }

        idx += 1;
        res.is_ok()
    });
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

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`. Note that
/// this is only inline(never) when backtraces in libstd are enabled, otherwise
/// it's fine to optimize away.
#[cfg_attr(feature = "backtrace", inline(never))]
pub fn __rust_begin_short_backtrace<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let result = f();

    // prevent this frame from being tail-call optimised away
    crate::hint::black_box(());

    result
}

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`. Note that
/// this is only inline(never) when backtraces in libstd are enabled, otherwise
/// it's fine to optimize away.
#[cfg_attr(feature = "backtrace", inline(never))]
pub fn __rust_end_short_backtrace<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let result = f();

    // prevent this frame from being tail-call optimised away
    crate::hint::black_box(());

    result
}

pub enum RustBacktrace {
    Print(PrintFmt),
    Disabled,
    RuntimeDisabled,
}

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn rust_backtrace_env() -> RustBacktrace {
    // If the `backtrace` feature of this crate isn't enabled quickly return
    // `None` so this can be constant propagated all over the place to turn
    // optimize away callers.
    if !cfg!(feature = "backtrace") {
        return RustBacktrace::Disabled;
    }

    // Setting environment variables for Fuchsia components isn't a standard
    // or easily supported workflow. For now, always display backtraces.
    if cfg!(target_os = "fuchsia") {
        return RustBacktrace::Print(PrintFmt::Full);
    }

    static ENABLED: atomic::AtomicIsize = atomic::AtomicIsize::new(0);
    match ENABLED.load(Ordering::SeqCst) {
        0 => {}
        1 => return RustBacktrace::RuntimeDisabled,
        2 => return RustBacktrace::Print(PrintFmt::Short),
        _ => return RustBacktrace::Print(PrintFmt::Full),
    }

    let (format, cache) = env::var_os("RUST_BACKTRACE")
        .map(|x| {
            if &x == "0" {
                (RustBacktrace::RuntimeDisabled, 1)
            } else if &x == "full" {
                (RustBacktrace::Print(PrintFmt::Full), 3)
            } else {
                (RustBacktrace::Print(PrintFmt::Short), 2)
            }
        })
        .unwrap_or((RustBacktrace::RuntimeDisabled, 1));
    ENABLED.store(cache, Ordering::SeqCst);
    format
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
                    return write!(fmt, ".{}{}", path::MAIN_SEPARATOR, s);
                }
            }
        }
    }
    fmt::Display::fmt(&file.display(), fmt)
}
