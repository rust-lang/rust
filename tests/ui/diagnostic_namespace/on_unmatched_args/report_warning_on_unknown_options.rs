//@ check-pass
#![feature(diagnostic_on_unmatched_args)]

#[diagnostic::on_unmatched_args(unsupported = "foo")]
//~^ WARN malformed `diagnostic::on_unmatched_args` attribute [malformed_diagnostic_attributes]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
