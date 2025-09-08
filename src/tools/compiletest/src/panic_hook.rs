use std::backtrace::{Backtrace, BacktraceStatus};
use std::cell::Cell;
use std::fmt::{Display, Write};
use std::panic::PanicHookInfo;
use std::sync::{Arc, LazyLock, Mutex};
use std::{env, mem, panic, thread};

type PanicHook = Box<dyn Fn(&PanicHookInfo<'_>) + Sync + Send + 'static>;
type CaptureBuf = Arc<Mutex<String>>;

thread_local!(
    static CAPTURE_BUF: Cell<Option<CaptureBuf>> = const { Cell::new(None) };
);

/// Installs a custom panic hook that will divert panic output to a thread-local
/// capture buffer, but only for threads that have a capture buffer set.
///
/// Otherwise, the custom hook delegates to a copy of the default panic hook.
pub(crate) fn install_panic_hook() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| custom_panic_hook(&default_hook, info)));
}

pub(crate) fn set_capture_buf(buf: CaptureBuf) {
    CAPTURE_BUF.set(Some(buf));
}

pub(crate) fn take_capture_buf() -> Option<CaptureBuf> {
    CAPTURE_BUF.take()
}

fn custom_panic_hook(default_hook: &PanicHook, info: &panic::PanicHookInfo<'_>) {
    // Temporarily taking the capture buffer means that if a panic occurs in
    // the subsequent code, that panic will fall back to the default hook.
    let Some(buf) = take_capture_buf() else {
        // There was no capture buffer, so delegate to the default hook.
        default_hook(info);
        return;
    };

    let mut out = buf.lock().unwrap_or_else(|e| e.into_inner());

    let thread = thread::current().name().unwrap_or("(test runner)").to_owned();
    let location = get_location(info);
    let payload = payload_as_str(info).unwrap_or("Box<dyn Any>");
    let backtrace = Backtrace::capture();

    writeln!(out, "\nthread '{thread}' panicked at {location}:\n{payload}").unwrap();
    match backtrace.status() {
        BacktraceStatus::Captured => {
            let bt = trim_backtrace(backtrace.to_string());
            write!(out, "stack backtrace:\n{bt}",).unwrap();
        }
        BacktraceStatus::Disabled => {
            writeln!(
                out,
                "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace",
            )
            .unwrap();
        }
        _ => {}
    }

    drop(out);
    set_capture_buf(buf);
}

fn get_location<'a>(info: &'a PanicHookInfo<'_>) -> &'a dyn Display {
    match info.location() {
        Some(location) => location,
        None => &"(unknown)",
    }
}

/// FIXME(Zalathar): Replace with `PanicHookInfo::payload_as_str` when that's
/// stable in beta.
fn payload_as_str<'a>(info: &'a PanicHookInfo<'_>) -> Option<&'a str> {
    let payload = info.payload();
    if let Some(s) = payload.downcast_ref::<&str>() {
        Some(s)
    } else if let Some(s) = payload.downcast_ref::<String>() {
        Some(s)
    } else {
        None
    }
}

fn rust_backtrace_full() -> bool {
    static RUST_BACKTRACE_FULL: LazyLock<bool> =
        LazyLock::new(|| matches!(env::var("RUST_BACKTRACE").as_deref(), Ok("full")));
    *RUST_BACKTRACE_FULL
}

/// On stable, short backtraces are only available to the default panic hook,
/// so if we want something similar we have to resort to string processing.
fn trim_backtrace(full_backtrace: String) -> String {
    if rust_backtrace_full() {
        return full_backtrace;
    }

    let mut buf = String::with_capacity(full_backtrace.len());
    // Don't print any frames until after the first `__rust_end_short_backtrace`.
    let mut on = false;
    // After the short-backtrace state is toggled, skip its associated "at" if present.
    let mut skip_next_at = false;

    let mut lines = full_backtrace.lines();
    while let Some(line) = lines.next() {
        if mem::replace(&mut skip_next_at, false) && line.trim_start().starts_with("at ") {
            continue;
        }

        if line.contains("__rust_end_short_backtrace") {
            on = true;
            skip_next_at = true;
            continue;
        }
        if line.contains("__rust_begin_short_backtrace") {
            on = false;
            skip_next_at = true;
            continue;
        }

        if on {
            writeln!(buf, "{line}").unwrap();
        }
    }

    writeln!(
        buf,
        "note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace."
    )
    .unwrap();

    buf
}
