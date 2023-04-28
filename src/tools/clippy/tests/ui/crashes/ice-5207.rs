// compile-flags: --cap-lints=warn
// ^ for https://github.com/rust-lang/rust-clippy/issues/10645

// Regression test for https://github.com/rust-lang/rust-clippy/issues/5207
#![warn(clippy::future_not_send)]
pub async fn bar<'a, T: 'a>(_: T) {}

fn main() {}
