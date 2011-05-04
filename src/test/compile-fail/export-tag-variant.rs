// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: unresolved name

mod foo {
  export x;

  fn x() {
  }

  tag y {
    y1;
  }
}

fn main() {
  auto z = foo.y1;
}
