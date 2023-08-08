//@compile-flags: --cap-lints=warn
// https://github.com/rust-lang/rust-clippy/issues/10645

#![warn(clippy::future_not_send)]
pub async fn bar<'a, T: 'a>(_: T) {}

fn main() {}
