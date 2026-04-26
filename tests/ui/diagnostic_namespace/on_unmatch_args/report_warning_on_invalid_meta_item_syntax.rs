//@ check-pass
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args = "foo"]
//~^ WARN malformed `diagnostic::on_unmatch_args` attribute [malformed_diagnostic_attributes]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
