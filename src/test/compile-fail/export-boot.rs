// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: unknown module item

// rustboot has a different error message than rustc
// this test can die with rustboot, or rustc's error can change

mod foo {
  export x;
  fn x(int y) {
    log y;
  }
  fn z(int y) {
    log y;
  }
}

fn main() {
  foo::z(10);
}
