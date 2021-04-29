use crate::fmt;
use crate::io::prelude::*;
use crate::sys::stdio::panic_output;

pub fn dumb_print(args: fmt::Arguments<'_>) {
    if let Some(mut out) = panic_output() {
        let _ = out.write_fmt(args);
    }
}
