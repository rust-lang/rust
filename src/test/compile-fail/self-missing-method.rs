// error-pattern:attempted field access on type fn
fn main() {

  obj foo() {
      fn m() {
          self();
      }
  }

  let a = foo;
  a.m();
}
