#![feature(diagnostic_on_missing_args)]

#[macro_export]
#[diagnostic::on_missing_args(
    message = "pair! is missing its second argument",
    label = "add the missing value here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}
