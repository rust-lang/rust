// error-pattern: unresolved name
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
