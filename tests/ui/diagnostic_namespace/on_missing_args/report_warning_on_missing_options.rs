//@ check-pass
#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args]
//~^ WARN missing options for `on_missing_args` attribute [malformed_diagnostic_attributes]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
