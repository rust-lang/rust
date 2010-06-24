// -*- rust -*-

fn f(str caller) {
  log caller;
}

fn main() {
  ret f("main");
  log "Paul is dead";
}

