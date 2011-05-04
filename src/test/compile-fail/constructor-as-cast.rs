// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: non-type context
type base =
  obj {
    fn foo();
  };
obj derived() {
  fn foo() {}
  fn bar() {}
}
fn main() {
  let derived d = derived();
  let base b = base(d);
}
