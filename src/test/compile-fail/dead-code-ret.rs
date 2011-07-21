// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-

// error-pattern: dead

fn f(str caller) {
  log caller;
}

fn main() {
  ret f("main");
  log "Paul is dead";
}

