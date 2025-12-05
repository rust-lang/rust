//@ check-pass
#[deprecated = concat !()]
macro_rules! a {
    () => {};
}
fn main() {}
