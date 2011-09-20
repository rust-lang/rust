// Issue #945
// error-pattern:non-exhaustive match failure
fn test_box() {
    @0;
}
fn test_str() {
    let res = alt false { true { "happy" } };
    assert res == "happy";
}
fn main() {
    test_box();
    test_str();
}