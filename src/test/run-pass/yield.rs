// -*- rust -*-

fn main() {
  auto other = spawn child();
  log "1";
  yield;
  log "2";
  yield;
  log "3";
  join other;
}

fn child() {
  log "4";
  yield;
  log "5";
  yield;
  log "6";
}

