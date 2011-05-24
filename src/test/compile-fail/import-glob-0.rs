mod module_of_many_things {
  export f1, f2, f4;
  fn f1() {
    log "f1";
  }
  fn f2() {
    log "f2";
  }
  fn f3() {
    log "f3";
  }
  fn f4() {
    log "f4";
  }
}

import module_of_many_things::*;

fn main() {
  f1();
  f2();
  f3();
  f4();
}