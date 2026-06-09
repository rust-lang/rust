//@ check-pass
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args(
    message = "{T}! is missing arguments",
    //~^ WARN this format argument is not allowed in `#[diagnostic::on_unmatch_args]`
    //~| NOTE only `This` is allowed as a format argument
    //~| NOTE remove this format argument
    //~| NOTE `#[warn(malformed_diagnostic_format_literals)]` (part of `#[warn(unknown_or_malformed_diagnostic_attributes)]`) on by default
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
