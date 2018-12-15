// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Common code for printing the backtrace in the same way across the different
/// supported platforms.

use env;
use io::prelude::*;
use io;
use path::{self, Path};
use ptr;
use rustc_demangle::demangle;
use str;
use sync::atomic::{self, Ordering};
use sys::mutex::Mutex;

pub use sys::backtrace::{
    unwind_backtrace,
    resolve_symname,
    foreach_symbol_fileline,
    BacktraceContext
};

#[cfg(target_pointer_width = "64")]
pub const HEX_WIDTH: usize = 18;

#[cfg(target_pointer_width = "32")]
pub const HEX_WIDTH: usize = 10;

/// Represents an item in the backtrace list. See `unwind_backtrace` for how
/// it is created.
#[derive(Debug, Copy, Clone)]
pub struct Frame {
    /// Exact address of the call that failed.
    pub exact_position: *const u8,
    /// Address of the enclosing function.
    pub symbol_addr: *const u8,
    /// Which inlined function is this frame referring to
    pub inline_context: u32,
}

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
        return Ok(())
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
    let mut frames = [Frame {
        exact_position: ptr::null(),
        symbol_addr: ptr::null(),
        inline_context: 0,
    }; MAX_NB_FRAMES];
    let (nb_frames, context) = unwind_backtrace(&mut frames)?;
    let (skipped_before, skipped_after) =
        filter_frames(&frames[..nb_frames], format, &context);
    if skipped_before + skipped_after > 0 {
        writeln!(w, "note: Some details are omitted, \
                     run with `RUST_BACKTRACE=full` for a verbose backtrace.")?;
    }
    writeln!(w, "stack backtrace:")?;

    let filtered_frames = &frames[..nb_frames - skipped_after];
    for (index, frame) in filtered_frames.iter().skip(skipped_before).enumerate() {
        resolve_symname(*frame, |symname| {
            output(w, index, *frame, symname, format)
        }, &context)?;
        let has_more_filenames = foreach_symbol_fileline(*frame, |file, line| {
            output_fileline(w, file, line, format)
        }, &context)?;
        if has_more_filenames {
            w.write_all(b" <... and possibly more>")?;
        }
    }

    Ok(())
}

/// Returns a number of frames to remove at the beginning and at the end of the
/// backtrace, according to the backtrace format.
fn filter_frames(frames: &[Frame],
                 format: PrintFormat,
                 context: &BacktraceContext) -> (usize, usize)
{
    if format == PrintFormat::Full {
        return (0, 0);
    }

    let skipped_before = 0;

    let skipped_after = frames.len() - frames.iter().position(|frame| {
        let mut is_marker = false;
        let _ = resolve_symname(*frame, |symname| {
            if let Some(mangled_symbol_name) = symname {
                // Use grep to find the concerned functions
                if mangled_symbol_name.contains("__rust_begin_short_backtrace") {
                    is_marker = true;
                }
            }
            Ok(())
        }, context);
        is_marker
    }).unwrap_or(frames.len());

    if skipped_before + skipped_after >= frames.len() {
        // Avoid showing completely empty backtraces
        return (0, 0);
    }

    (skipped_before, skipped_after)
}


/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`.
#[inline(never)]
pub fn __rust_begin_short_backtrace<F, T>(f: F) -> T
    where F: FnOnce() -> T, F: Send, T: Send
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

    let val = env::var_os("RUST_BACKTRACE").and_then(|x|
        if &x == "0" {
            None
        } else if &x == "full" {
            Some(PrintFormat::Full)
        } else {
            Some(PrintFormat::Short)
        }
    );
    ENABLED.store(match val {
        Some(v) => v as isize,
        None => 1,
    }, Ordering::SeqCst);
    val
}

/// Print the symbol of the backtrace frame.
///
/// These output functions should now be used everywhere to ensure consistency.
/// You may want to also use `output_fileline`.
fn output(w: &mut dyn Write, idx: usize, frame: Frame,
              s: Option<&str>, format: PrintFormat) -> io::Result<()> {
    // Remove the `17: 0x0 - <unknown>` line.
    if format == PrintFormat::Short && frame.exact_position == ptr::null() {
        return Ok(());
    }
    match format {
        PrintFormat::Full => write!(w,
                                    "  {:2}: {:2$?} - ",
                                    idx,
                                    frame.exact_position,
                                    HEX_WIDTH)?,
        PrintFormat::Short => write!(w, "  {:2}: ", idx)?,
    }
    match s {
        Some(string) => {
            let symbol = demangle(string);
            match format {
                PrintFormat::Full => write!(w, "{}", symbol)?,
                // strip the trailing hash if short mode
                PrintFormat::Short => write!(w, "{:#}", symbol)?,
            }
        }
        None => w.write_all(b"<unknown>")?,
    }
    w.write_all(b"\n")
}

/// Print the filename and line number of the backtrace frame.
///
/// See also `output`.
#[allow(dead_code)]
fn output_fileline(w: &mut dyn Write,
                   file: &[u8],
                   line: u32,
                   format: PrintFormat) -> io::Result<()> {
    // prior line: "  ##: {:2$} - func"
    w.write_all(b"")?;
    match format {
        PrintFormat::Full => write!(w,
                                    "           {:1$}",
                                    "",
                                    HEX_WIDTH)?,
        PrintFormat::Short => write!(w, "           ")?,
    }

    let file = str::from_utf8(file).unwrap_or("<unknown>");
    let file_path = Path::new(file);
    let mut already_printed = false;
    if format == PrintFormat::Short && file_path.is_absolute() {
        if let Ok(cwd) = env::current_dir() {
            if let Ok(stripped) = file_path.strip_prefix(&cwd) {
                if let Some(s) = stripped.to_str() {
                    write!(w, "  at .{}{}:{}", path::MAIN_SEPARATOR, s, line)?;
                    already_printed = true;
                }
            }
        }
    }
    if !already_printed {
        write!(w, "  at {}:{}", file, line)?;
    }

    w.write_all(b"\n")
}

