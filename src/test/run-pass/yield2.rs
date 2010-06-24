// -*- rust -*-

fn main() {
  let int i = 0;
  while (i < 100) {
    i = i + 1;
    log i;
    yield;
  }
}
