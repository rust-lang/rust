// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: unresolved name

mod foo {
  export x;

  fn x() {
    bar.x();
  }
}

mod bar {
  export y;

  fn x() {
    log "x";
  }

  fn y() {
  }
}

fn main() {
  foo.x();
}
