// This is a test for issue #109.

use std;

fn slice[T](vec[T] e) {
  let vec[T] result = std::vec::alloc[T](1 as uint);
  log "alloced";
  result += e;
  log "appended";
}

fn main() {
  slice[str](["a"]);
}
