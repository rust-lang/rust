use std::fmt;
use std::io::{self, Write as _};

macro_rules! safe_print {
    ($($arg:tt)*) => {{
        $crate::print::print(std::format_args!($($arg)*));
    }};
}

macro_rules! safe_println {
    ($($arg:tt)*) => {
        safe_print!("{}\n", std::format_args!($($arg)*))
    };
}

pub(crate) fn print(args: fmt::Arguments<'_>) {
    if io::stdout().write_fmt(args).is_err() {
        rustc_errors::FatalError.raise();
    }
}
