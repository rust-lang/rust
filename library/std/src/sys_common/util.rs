use crate::fmt;
use crate::io::prelude::*;
use crate::sys::stdio::panic_output;
use crate::thread;

pub fn dumb_print(args: fmt::Arguments<'_>) {
    if let Some(mut out) = panic_output() {
        let _ = out.write_fmt(args);
    }
}

#[allow(dead_code)] // stack overflow detection not enabled on all platforms
pub unsafe fn report_overflow() {
    dumb_print(format_args!(
        "\nthread '{}' has overflowed its stack\n",
        thread::current().name().unwrap_or("<unknown>")
    ));
}
