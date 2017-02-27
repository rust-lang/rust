// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(target_os = "nacl", allow(dead_code))]

/// Common code for printing the backtrace in the same way across the different
/// supported platforms.

use env;
use io::prelude::*;
use io;
use libc;
use str;
use sync::atomic::{self, Ordering};
use path::Path;
use sys::mutex::Mutex;
use ptr;

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
    pub exact_position: *const libc::c_void,
    /// Address of the enclosing function.
    pub symbol_addr: *const libc::c_void,
}

/// Max number of frames to print.
const MAX_NB_FRAMES: usize = 100;

/// Prints the current backtrace.
pub fn print(w: &mut Write, format: PrintFormat) -> io::Result<()> {
    static LOCK: Mutex = Mutex::new();

    // Use a lock to prevent mixed output in multithreading context.
    // Some platforms also requires it, like `SymFromAddr` on Windows.
    unsafe {
        LOCK.lock();
        let res = _print(w, format);
        LOCK.unlock();
        res
    }
}

fn _print(w: &mut Write, format: PrintFormat) -> io::Result<()> {
    let mut frames = [Frame {
        exact_position: ptr::null(),
        symbol_addr: ptr::null(),
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

fn filter_frames(frames: &[Frame],
                 format: PrintFormat,
                 context: &BacktraceContext) -> (usize, usize)
{
    if format == PrintFormat::Full {
        return (0, 0);
    }

    // We want to filter out frames with some prefixes
    // from both top and bottom of the call stack.
    static BAD_PREFIXES_TOP: &'static [&'static str] = &[
        "_ZN3std3sys3imp9backtrace",
        "ZN3std3sys3imp9backtrace",
        "std::sys::imp::backtrace",
        "_ZN3std10sys_common9backtrace",
        "ZN3std10sys_common9backtrace",
        "std::sys_common::backtrace",
        "_ZN3std9panicking",
        "ZN3std9panicking",
        "std::panicking",
        "_ZN4core9panicking",
        "ZN4core9panicking",
        "core::panicking",
        "_ZN4core6result13unwrap_failed",
        "ZN4core6result13unwrap_failed",
        "core::result::unwrap_failed",
        "rust_begin_unwind",
        "_ZN4drop",
        "mingw_set_invalid_parameter_handler",
    ];
    static BAD_PREFIXES_BOTTOM: &'static [&'static str] = &[
        "_ZN3std9panicking",
        "ZN3std9panicking",
        "std::panicking",
        "_ZN3std5panic",
        "ZN3std5panic",
        "std::panic",
        "_ZN4core9panicking",
        "ZN4core9panicking",
        "core::panicking",
        "_ZN3std2rt10lang_start",
        "ZN3std2rt10lang_start",
        "std::rt::lang_start",
        "panic_unwind::__rust_maybe_catch_panic",
        "__rust_maybe_catch_panic",
        "_rust_maybe_catch_panic",
        "__libc_start_main",
        "__rust_try",
        "_start",
        "main",
        "BaseThreadInitThunk",
        "RtlInitializeExceptionChain",
        "__scrt_common_main_seh",
        "_ZN4drop",
        "mingw_set_invalid_parameter_handler",
    ];

    let is_good_frame = |frame: Frame, bad_prefixes: &[&str]| {
        resolve_symname(frame, |symname| {
            if let Some(mangled_symbol_name) = symname {
                if !bad_prefixes.iter().any(|s| mangled_symbol_name.starts_with(s)) {
                    return Ok(())
                }
            }
            Err(io::Error::from(io::ErrorKind::Other))
        }, context).is_ok()
    };

    let skipped_before = frames.iter().position(|frame| {
        is_good_frame(*frame, BAD_PREFIXES_TOP)
    }).unwrap_or(frames.len());
    let skipped_after = frames[skipped_before..].iter().rev().position(|frame| {
        is_good_frame(*frame, BAD_PREFIXES_BOTTOM)
    }).unwrap_or(frames.len() - skipped_before);

    if skipped_before + skipped_after == frames.len() {
        // Avoid showing completely empty backtraces
        return (0, 0);
    }

    (skipped_before, skipped_after)
}

/// Controls how the backtrace should be formated.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrintFormat {
    /// Show all the frames with absolute path for files.
    Full = 2,
    /// Show only relevant data from the backtrace.
    Short = 3,
}

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn log_enabled() -> Option<PrintFormat> {
    static ENABLED: atomic::AtomicIsize = atomic::AtomicIsize::new(0);
    match ENABLED.load(Ordering::SeqCst) {
        0 => {},
        1 => return None,
        2 => return Some(PrintFormat::Full),
        3 => return Some(PrintFormat::Short),
        _ => unreachable!(),
    }

    let val = match env::var_os("RUST_BACKTRACE") {
        Some(x) => if &x == "0" {
            None
        } else if &x == "full" {
            Some(PrintFormat::Full)
        } else {
            Some(PrintFormat::Short)
        },
        None => None,
    };
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
fn output(w: &mut Write, idx: usize, frame: Frame,
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
        Some(string) => demangle(w, string, format)?,
        None => w.write_all(b"<unknown>")?,
    }
    w.write_all(b"\n")
}

/// Print the filename and line number of the backtrace frame.
///
/// See also `output`.
#[allow(dead_code)]
fn output_fileline(w: &mut Write, file: &[u8], line: libc::c_int,
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
                    write!(w, "  at ./{}:{}", s, line)?;
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


// All rust symbols are in theory lists of "::"-separated identifiers. Some
// assemblers, however, can't handle these characters in symbol names. To get
// around this, we use C++-style mangling. The mangling method is:
//
// 1. Prefix the symbol with "_ZN"
// 2. For each element of the path, emit the length plus the element
// 3. End the path with "E"
//
// For example, "_ZN4testE" => "test" and "_ZN3foo3barE" => "foo::bar".
//
// We're the ones printing our backtraces, so we can't rely on anything else to
// demangle our symbols. It's *much* nicer to look at demangled symbols, so
// this function is implemented to give us nice pretty output.
//
// Note that this demangler isn't quite as fancy as it could be. We have lots
// of other information in our symbols like hashes, version, type information,
// etc. Additionally, this doesn't handle glue symbols at all.
pub fn demangle(writer: &mut Write, s: &str, format: PrintFormat) -> io::Result<()> {
    // First validate the symbol. If it doesn't look like anything we're
    // expecting, we just print it literally. Note that we must handle non-rust
    // symbols because we could have any function in the backtrace.
    let mut valid = true;
    let mut inner = s;
    if s.len() > 4 && s.starts_with("_ZN") && s.ends_with("E") {
        inner = &s[3 .. s.len() - 1];
    // On Windows, dbghelp strips leading underscores, so we accept "ZN...E" form too.
    } else if s.len() > 3 && s.starts_with("ZN") && s.ends_with("E") {
        inner = &s[2 .. s.len() - 1];
    } else {
        valid = false;
    }

    if valid {
        let mut chars = inner.chars();
        while valid {
            let mut i = 0;
            for c in chars.by_ref() {
                if c.is_numeric() {
                    i = i * 10 + c as usize - '0' as usize;
                } else {
                    break
                }
            }
            if i == 0 {
                valid = chars.next().is_none();
                break
            } else if chars.by_ref().take(i - 1).count() != i - 1 {
                valid = false;
            }
        }
    }

    // Alright, let's do this.
    if !valid {
        writer.write_all(s.as_bytes())?;
    } else {
        // remove the `::hfc2edb670e5eda97` part at the end of the symbol.
        if format == PrintFormat::Short {
            // The symbol in still mangled.
            let mut split = inner.rsplitn(2, "17h");
            match (split.next(), split.next()) {
                (Some(addr), rest) => {
                    if addr.len() == 16 &&
                       addr.chars().all(|c| c.is_digit(16))
                    {
                        inner = rest.unwrap_or("");
                    }
                }
                _ => (),
            }
        }

        let mut first = true;
        while !inner.is_empty() {
            if !first {
                writer.write_all(b"::")?;
            } else {
                first = false;
            }
            let mut rest = inner;
            while rest.chars().next().unwrap().is_numeric() {
                rest = &rest[1..];
            }
            let i: usize = inner[.. (inner.len() - rest.len())].parse().unwrap();
            inner = &rest[i..];
            rest = &rest[..i];
            if rest.starts_with("_$") {
                rest = &rest[1..];
            }
            while !rest.is_empty() {
                if rest.starts_with(".") {
                    if let Some('.') = rest[1..].chars().next() {
                        writer.write_all(b"::")?;
                        rest = &rest[2..];
                    } else {
                        writer.write_all(b".")?;
                        rest = &rest[1..];
                    }
                } else if rest.starts_with("$") {
                    macro_rules! demangle {
                        ($($pat:expr => $demangled:expr),*) => ({
                            $(if rest.starts_with($pat) {
                                writer.write_all($demangled)?;
                                rest = &rest[$pat.len()..];
                              } else)*
                            {
                                writer.write_all(rest.as_bytes())?;
                                break;
                            }

                        })
                    }

                    // see src/librustc/back/link.rs for these mappings
                    demangle! (
                        "$SP$" => b"@",
                        "$BP$" => b"*",
                        "$RF$" => b"&",
                        "$LT$" => b"<",
                        "$GT$" => b">",
                        "$LP$" => b"(",
                        "$RP$" => b")",
                        "$C$" => b",",

                        // in theory we can demangle any Unicode code point, but
                        // for simplicity we just catch the common ones.
                        "$u7e$" => b"~",
                        "$u20$" => b" ",
                        "$u27$" => b"'",
                        "$u5b$" => b"[",
                        "$u5d$" => b"]",
                        "$u7b$" => b"{",
                        "$u7d$" => b"}",
                        "$u3b$" => b";",
                        "$u2b$" => b"+",
                        "$u22$" => b"\""
                    )
                } else {
                    let idx = match rest.char_indices().find(|&(_, c)| c == '$' || c == '.') {
                        None => rest.len(),
                        Some((i, _)) => i,
                    };
                    writer.write_all(rest[..idx].as_bytes())?;
                    rest = &rest[idx..];
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use sys_common;
    macro_rules! t { ($a:expr, $b:expr) => ({
        let mut m = Vec::new();
        sys_common::backtrace::demangle(&mut m,
                                        $a,
                                        super::PrintFormat::Full).unwrap();
        assert_eq!(String::from_utf8(m).unwrap(), $b);
    }) }

    #[test]
    fn demangle() {
        t!("test", "test");
        t!("_ZN4testE", "test");
        t!("_ZN4test", "_ZN4test");
        t!("_ZN4test1a2bcE", "test::a::bc");
    }

    #[test]
    fn demangle_dollars() {
        t!("_ZN4$RP$E", ")");
        t!("_ZN8$RF$testE", "&test");
        t!("_ZN8$BP$test4foobE", "*test::foob");
        t!("_ZN9$u20$test4foobE", " test::foob");
        t!("_ZN35Bar$LT$$u5b$u32$u3b$$u20$4$u5d$$GT$E", "Bar<[u32; 4]>");
    }

    #[test]
    fn demangle_many_dollars() {
        t!("_ZN13test$u20$test4foobE", "test test::foob");
        t!("_ZN12test$BP$test4foobE", "test*test::foob");
    }

    #[test]
    fn demangle_windows() {
        t!("ZN4testE", "test");
        t!("ZN13test$u20$test4foobE", "test test::foob");
        t!("ZN12test$RF$test4foobE", "test&test::foob");
    }

    #[test]
    fn demangle_elements_beginning_with_underscore() {
        t!("_ZN13_$LT$test$GT$E", "<test>");
        t!("_ZN28_$u7b$$u7b$closure$u7d$$u7d$E", "{{closure}}");
        t!("_ZN15__STATIC_FMTSTRE", "__STATIC_FMTSTR");
    }

    #[test]
    fn demangle_trait_impls() {
        t!("_ZN71_$LT$Test$u20$$u2b$$u20$$u27$static$u20$as$u20$foo..Bar$LT$Test$GT$$GT$3barE",
           "<Test + 'static as foo::Bar<Test>>::bar");
    }
}
