#![crate_type = "dylib"]

pub fn print(_args: std::fmt::Arguments) {}

#[macro_export]
macro_rules! myprint {
    ($($arg:tt)*) => (print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! myprintln {
    ($fmt:expr) => (myprint!(concat!($fmt, "\n")));
}
