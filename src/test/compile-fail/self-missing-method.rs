// error-pattern:expecting ., found (
fn main() {

  obj foo() {
      fn m() {
          self();
      }
  }

  let a = foo;
  a.m();
}
