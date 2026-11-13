//! Regression test for https://github.com/rust-lang/rust/issues/18159
fn main() {
    let x; //~ ERROR type annotations needed
}
