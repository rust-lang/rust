// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: unresolved name

import m.unexported;

mod m {
  export exported;

  fn exported() {
  }

  fn unexported() {
  }
}


fn main() {
  unexported();
}
