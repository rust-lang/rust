// xfail-test
trait A {
  fn a(&self) {
    || self.b()
  }
}
