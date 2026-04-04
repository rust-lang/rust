//@ check-pass
#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args(
    message = "pair! is missing {T}",
    //~^ WARN unknown parameter `T`
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
