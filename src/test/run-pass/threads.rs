// -*- rust -*-

fn main() {
  let int i = 10;
  while (i > 0) {
    auto child_task = spawn thread "child" child(i);
    i = i - 1;
  }
  log "main thread exiting";
}

fn child(int x) {
  log x;
}

