//xfail-stage1
//xfail-stage2
//xfail-stage3

// Test case for issue #115.
type base =
  obj {
    fn foo();
  };

obj derived() {
  fn foo() {}
  fn bar() {}
}

fn main() {
  let d = derived();
  let b:base = d as base;
}
