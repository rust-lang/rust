// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: unresolved name
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
  foo.z(10);
}
