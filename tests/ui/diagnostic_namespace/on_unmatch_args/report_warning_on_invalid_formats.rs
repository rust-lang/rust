//@ check-pass
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args(
    message = "{T}! is missing arguments",
    //~^ WARN unknown parameter `T`
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

fn main() {
    pair!(u8, 0);
}
