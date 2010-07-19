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
