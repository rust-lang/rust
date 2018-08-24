// The `format_args!` syntax extension issues errors before code expansion
// has completed, but we still need a backtrace.

// This test includes stripped-down versions of `print!` and `println!`,
// because we can't otherwise verify the lines of the backtrace.

fn print(_args: std::fmt::Arguments) {}

macro_rules! myprint {
    ($($arg:tt)*) => (print(format_args!($($arg)*)));
}

macro_rules! myprintln {
    ($fmt:expr) => (myprint!(concat!($fmt, "\n"))); //~ ERROR no arguments were given
}

fn main() {
    myprintln!("{}");
}
