// error-pattern:attempted access of field m on type fn
fn main() {

  obj foo() {
      fn m() {
          self();
      }
  }

  let a = foo;
  a.m();
}
