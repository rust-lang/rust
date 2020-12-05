#![crate_type = "lib"]

#[macro_export]
macro_rules! mywrite {
    ($dst:expr, $($arg:tt)*) => ($dst.write_fmt(format_args!($($arg)*)))
}
