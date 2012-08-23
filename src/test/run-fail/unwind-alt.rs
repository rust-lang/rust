// Issue #945
// error-pattern:non-exhaustive match failure
fn test_box() {
    @0;
}
fn test_str() {
  let res = match false { true => { ~"happy" },
     _ => fail ~"non-exhaustive match failure" };
  assert res == ~"happy";
}
fn main() {
    test_box();
    test_str();
}