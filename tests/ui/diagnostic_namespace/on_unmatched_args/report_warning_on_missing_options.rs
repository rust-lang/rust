//@ check-pass
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatched_args]
//~^ WARN missing options for `diagnostic::on_unmatched_args` attribute [malformed_diagnostic_attributes]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
