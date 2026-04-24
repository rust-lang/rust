#![feature(diagnostic_on_unmatch_args)]

#[macro_export]
#[diagnostic::on_unmatch_args(
    message = "invalid arguments to {This} macro invocation",
    label = "expected a type and value here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
    note = "see the macro documentation for accepted forms",
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}
