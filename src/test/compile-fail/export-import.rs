// xfail-boot
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
