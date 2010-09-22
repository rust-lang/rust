// -*- rust -*-

fn x(str s, int n) {
  log s;
  log n;
}

fn main() {
  auto task1 = spawn x("hello from first spawned fn", 65);
  auto task2 = spawn x("hello from second spawned fn", 66);
  auto task3 = spawn x("hello from third spawned fn", 67);
  let int i = 30;
  while (i > 0) {
    i = i - 1;
    log "parent sleeping";
    yield;
  }
}
