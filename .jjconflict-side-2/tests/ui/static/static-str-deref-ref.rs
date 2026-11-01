//! regression test for <https://github.com/rust-lang/rust/issues/22894>
//@ build-pass
#[allow(dead_code)]
static X: &'static str = &*"";
fn main() {}
