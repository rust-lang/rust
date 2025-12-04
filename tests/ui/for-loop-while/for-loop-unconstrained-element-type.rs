// Test that `for` loops don't introduce artificial
// constraints on the type of the binding (`i`).
// Subtle changes in the desugaring can cause the
// type of elements in the vector to (incorrectly)
// fallback to `!` or `()`.
// regression test for issue <https://github.com/rust-lang/rust/issues/42618>

fn main() {
    for i in Vec::new() {} //~ ERROR type annotations needed
}
