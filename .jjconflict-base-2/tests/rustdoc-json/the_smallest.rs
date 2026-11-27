// This test asserts that `index` is not polluted with unrelated items.
// See https://github.com/rust-lang/rust/issues/114039

//@ count "$.index[*]" 1
fn main() {}
