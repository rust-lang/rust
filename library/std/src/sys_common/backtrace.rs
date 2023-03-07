use crate::backtrace_rs::{self, BacktraceFmt, BytesOrWideString, PrintFmt};
use crate::borrow::Cow;
/// Common code for printing the backtrace in the same way across the different
/// supported platforms.
use crate::env;
use crate::fmt;
use crate::io;
use crate::io::prelude::*;
use crate::path::{self, Path, PathBuf};
use crate::sync::{Mutex, PoisonError};

/// Max number of frames to print.
const MAX_NB_FRAMES: usize = 100;

pub fn lock() -> impl Drop {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Prints the current backtrace.
pub fn print(w: &mut dyn Write, format: PrintFmt) -> io::Result<()> {
    // There are issues currently linking libbacktrace into tests, and in
    // general during std's own unit tests we're not testing this path. In
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
        #[cfg(target_os = "nto")]
        if libc::__my_thread_exit as *mut libc::c_void == frame.ip() {
            if !hit && start {
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
/// this is only inline(never) when backtraces in std are enabled, otherwise
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
/// this is only inline(never) when backtraces in std are enabled, otherwise
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
