// error-pattern:expecting ., found (
fn main() {

  obj foo() {
      fn m() {
          self();
      }
  }

  auto a = foo;
  a.m();
}
