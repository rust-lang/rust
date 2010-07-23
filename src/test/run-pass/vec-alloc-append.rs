// This is a test for issue #109.

use std;

fn slice[T](vec[T] e) {
  let vec[T] result = std._vec.alloc[T](uint(1));
  log "alloced";
  result += e;
  log "appended";
}

fn main() {
  slice[str](vec("a"));
}
