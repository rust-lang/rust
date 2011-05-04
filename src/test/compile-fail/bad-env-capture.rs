// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: attempted dynamic environment-capture
fn foo() {
  let int x;
  fn bar() {
    log x;
  }
}
fn main() {
  foo();
}