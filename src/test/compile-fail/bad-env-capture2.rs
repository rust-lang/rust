// error-pattern: attempted dynamic environment-capture
fn foo(int x) {
  fn bar() {
    log x;
  }
}
fn main() {
  foo(2);
}