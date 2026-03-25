use std::hint::{likely, unlikely};
use std::io::{self, Write};
use std::sync::atomic::{AtomicU16, Ordering};

use backtrace::{Backtrace, BacktraceFrame};

// This is the amount of bytes that need to be left on the stack before increasing the size.
// It must be at least as large as the stack required by any code that does not call
// `ensure_sufficient_stack`.
const RED_ZONE: usize = 100 * 1024; // 100k

// Only the first stack that is pushed, grows exponentially (2^n * STACK_PER_RECURSION) from then
// on. This flag has performance relevant characteristics. Don't set it too high.
#[cfg(not(target_os = "aix"))]
const STACK_PER_RECURSION: usize = 1024 * 1024; // 1MB
// LLVM for AIX doesn't feature TCO, increase recursion size for workaround.
#[cfg(target_os = "aix")]
const STACK_PER_RECURSION: usize = 16 * 1024 * 1024; // 16MB

thread_local! {
    static TIMES_GROWN: AtomicU16 = const { AtomicU16::new(0) };
}

// Give up if we expand the stack this many times and are still trying to recurse deeper.
const MAX_STACK_GROWTH: u16 = 1000;

/// Grows the stack on demand to prevent stack overflow. Call this in strategic locations
/// to "break up" recursive calls. E.g. almost any call to `visit_expr` or equivalent can benefit
/// from this.
///
/// Should not be sprinkled around carelessly, as it causes a little bit of overhead.
pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
    // if we can't guess the remaining stack (unsupported on some platforms) we immediately grow
    // the stack and then cache the new stack size (which we do know now because we allocated it.
    let enough_space = match stacker::remaining_stack() {
        Some(remaining) => remaining >= RED_ZONE,
        None => false,
    };
    if likely(enough_space) {
        f()
    } else {
        let times = TIMES_GROWN.with(|times| times.fetch_add(1, Ordering::Relaxed));
        if unlikely(times > MAX_STACK_GROWTH) {
            too_much_stack();
        }
        let out = stacker::grow(STACK_PER_RECURSION, f);
        TIMES_GROWN.with(|times| times.fetch_sub(1, Ordering::Relaxed));
        out
    }
}

#[cold]
fn too_much_stack() -> ! {
    let Err(e) = std::panic::catch_unwind(report_too_much_stack);
    let mut stderr = io::stderr();
    let _ = writeln!(stderr, "ensure_sufficient_stack: panicked while handling stack overflow!");
    if let Ok(s) = e.downcast::<String>() {
        let _ = writeln!(stderr, "{s}");
    }
    std::process::abort();
}

#[cold]
fn report_too_much_stack() -> ! {
    // something is *definitely* wrong.
    eprintln!(
        "still not enough stack after {MAX_STACK_GROWTH} expansions of dynamic stack; infinite recursion?"
    );

    let backtrace = Backtrace::new_unresolved();
    let frames = backtrace.frames();
    eprintln!("first hundred frames:");
    print_frames(0, &frames[..100]);

    eprintln!("...\nlast hundred frames:");
    let start = frames.len() - 100;
    print_frames(start, &frames[start..]);
    std::process::abort();
}

#[cold]
fn print_frames(mut i: usize, frames: &[BacktraceFrame]) {
    for frame in frames {
        let mut frame = frame.clone();
        frame.resolve();
        for symbol in frame.symbols() {
            eprint!("{i}: ");
            match symbol.name() {
                Some(sym) => eprint!("{sym}"),
                None => eprint!("<unknown>"),
            }
            eprint!("\n\t\tat ");
            if let Some(file) = symbol.filename() {
                eprint!("{}", file.display());
                if let Some(line) = symbol.lineno() {
                    eprint!(":{line}");
                    if let Some(col) = symbol.colno() {
                        eprint!(":{col}");
                    }
                }
            } else {
                eprint!("<unknown>");
            }
            eprintln!();
            i += 1;
        }
    }
}
