// error-pattern:giraffe
fn main() {
  fail do { fail "giraffe" } while true;
}
