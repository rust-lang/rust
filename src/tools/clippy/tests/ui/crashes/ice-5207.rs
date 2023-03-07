// Regression test for https://github.com/rust-lang/rust-clippy/issues/5207

pub async fn bar<'a, T: 'a>(_: T) {}

fn main() {}
