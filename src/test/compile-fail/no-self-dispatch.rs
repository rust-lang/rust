// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: unresolved identifier
obj oT() {
  fn get() -> int {
    ret 3;
  }
  fn foo() {
    auto c = get();
  }
}
fn main() {
}
