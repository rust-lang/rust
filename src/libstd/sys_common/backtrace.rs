/// Common code for printing the backtrace in the same way across the different
/// supported platforms.

use crate::env;
use crate::io;
use crate::io::prelude::*;
use crate::mem;
use crate::path::{self, Path};
use crate::ptr;
use crate::sync::atomic::{self, Ordering};
use crate::sys::mutex::Mutex;

use backtrace::{BytesOrWideString, Frame, Symbol};

pub const HEX_WIDTH: usize = 2 + 2 * mem::size_of::<usize>();

/// Max number of frames to print.
const MAX_NB_FRAMES: usize = 100;

/// Prints the current backtrace.
pub fn print(w: &mut dyn Write, format: PrintFormat) -> io::Result<()> {
    static LOCK: Mutex = Mutex::new();

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
        LOCK.lock();
        let res = _print(w, format);
        LOCK.unlock();
        res
    }
}

fn _print(w: &mut dyn Write, format: PrintFormat) -> io::Result<()> {
    writeln!(w, "stack backtrace:")?;

    let mut printer = Printer::new(format, w);
    unsafe {
        backtrace::trace_unsynchronized(|frame| {
            let mut hit = false;
            backtrace::resolve_frame_unsynchronized(frame, |symbol| {
                hit = true;
                printer.output(frame, Some(symbol));
            });
            if !hit {
                printer.output(frame, None);
            }
            !printer.done
        });
    }
    if printer.skipped {
        writeln!(
            w,
            "note: Some details are omitted, \
             run with `RUST_BACKTRACE=full` for a verbose backtrace."
        )?;
    }
    Ok(())
}

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`.
#[inline(never)]
pub fn __rust_begin_short_backtrace<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
    F: Send,
    T: Send,
{
    f()
}

/// Controls how the backtrace should be formatted.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrintFormat {
    /// Show only relevant data from the backtrace.
    Short = 2,
    /// Show all the frames with absolute path for files.
    Full = 3,
}

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn log_enabled() -> Option<PrintFormat> {
    static ENABLED: atomic::AtomicIsize = atomic::AtomicIsize::new(0);
    match ENABLED.load(Ordering::SeqCst) {
        0 => {}
        1 => return None,
        2 => return Some(PrintFormat::Short),
        _ => return Some(PrintFormat::Full),
    }

    let val = env::var_os("RUST_BACKTRACE").and_then(|x| {
        if &x == "0" {
            None
        } else if &x == "full" {
            Some(PrintFormat::Full)
        } else {
            Some(PrintFormat::Short)
        }
    });
    ENABLED.store(
        match val {
            Some(v) => v as isize,
            None => 1,
        },
        Ordering::SeqCst,
    );
    val
}

struct Printer<'a, 'b> {
    format: PrintFormat,
    done: bool,
    skipped: bool,
    idx: usize,
    out: &'a mut (dyn Write + 'b),
}

impl<'a, 'b> Printer<'a, 'b> {
    fn new(format: PrintFormat, out: &'a mut (dyn Write + 'b)) -> Printer<'a, 'b> {
        Printer { format, done: false, skipped: false, idx: 0, out }
    }

    /// Prints the symbol of the backtrace frame.
    ///
    /// These output functions should now be used everywhere to ensure consistency.
    /// You may want to also use `output_fileline`.
    fn output(&mut self, frame: &Frame, symbol: Option<&Symbol>) {
        if self.idx > MAX_NB_FRAMES {
            self.done = true;
            self.skipped = true;
            return;
        }
        if self._output(frame, symbol).is_err() {
            self.done = true;
        }
        self.idx += 1;
    }

    fn _output(&mut self, frame: &Frame, symbol: Option<&Symbol>) -> io::Result<()> {
        if self.format == PrintFormat::Short {
            if let Some(sym) = symbol.and_then(|s| s.name()).and_then(|s| s.as_str()) {
                if sym.contains("__rust_begin_short_backtrace") {
                    self.skipped = true;
                    self.done = true;
                    return Ok(());
                }
            }

            // Remove the `17: 0x0 - <unknown>` line.
            if self.format == PrintFormat::Short && frame.ip() == ptr::null_mut() {
                self.skipped = true;
                return Ok(());
            }
        }

        match self.format {
            PrintFormat::Full => {
                write!(self.out, "  {:2}: {:2$?} - ", self.idx, frame.ip(), HEX_WIDTH)?
            }
            PrintFormat::Short => write!(self.out, "  {:2}: ", self.idx)?,
        }

        match symbol.and_then(|s| s.name()) {
            Some(symbol) => {
                match self.format {
                    PrintFormat::Full => write!(self.out, "{}", symbol)?,
                    // Strip the trailing hash if short mode.
                    PrintFormat::Short => write!(self.out, "{:#}", symbol)?,
                }
            }
            None => self.out.write_all(b"<unknown>")?,
        }
        self.out.write_all(b"\n")?;
        if let Some(sym) = symbol {
            self.output_fileline(sym)?;
        }
        Ok(())
    }

    /// Prints the filename and line number of the backtrace frame.
    ///
    /// See also `output`.
    fn output_fileline(&mut self, symbol: &Symbol) -> io::Result<()> {
        #[cfg(windows)]
        let path_buf;
        let file = match symbol.filename_raw() {
            #[cfg(unix)]
            Some(BytesOrWideString::Bytes(bytes)) => {
                use crate::os::unix::prelude::*;
                Path::new(crate::ffi::OsStr::from_bytes(bytes))
            }
            #[cfg(not(unix))]
            Some(BytesOrWideString::Bytes(bytes)) => {
                Path::new(crate::str::from_utf8(bytes).unwrap_or("<unknown>"))
            }
            #[cfg(windows)]
            Some(BytesOrWideString::Wide(wide)) => {
                use crate::os::windows::prelude::*;
                path_buf = crate::ffi::OsString::from_wide(wide);
                Path::new(&path_buf)
            }
            #[cfg(not(windows))]
            Some(BytesOrWideString::Wide(_wide)) => {
                Path::new("<unknown>")
            }
            None => return Ok(()),
        };
        let line = match symbol.lineno() {
            Some(line) => line,
            None => return Ok(()),
        };
        // prior line: "  ##: {:2$} - func"
        self.out.write_all(b"")?;
        match self.format {
            PrintFormat::Full => write!(self.out, "           {:1$}", "", HEX_WIDTH)?,
            PrintFormat::Short => write!(self.out, "           ")?,
        }

        let mut already_printed = false;
        if self.format == PrintFormat::Short && file.is_absolute() {
            if let Ok(cwd) = env::current_dir() {
                if let Ok(stripped) = file.strip_prefix(&cwd) {
                    if let Some(s) = stripped.to_str() {
                        write!(self.out, "  at .{}{}:{}", path::MAIN_SEPARATOR, s, line)?;
                        already_printed = true;
                    }
                }
            }
        }
        if !already_printed {
            write!(self.out, "  at {}:{}", file.display(), line)?;
        }

        self.out.write_all(b"\n")
    }
}
