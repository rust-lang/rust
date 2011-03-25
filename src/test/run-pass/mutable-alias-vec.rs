// xfail-stage0
// -*- rust -*-

use std;

fn grow(&mutable vec[int] v) {
  v += vec(1);
}

fn main() {
  let vec[int] v = vec();
  grow(v);
  grow(v);
  grow(v);
  auto len = std._vec.len[int](v);
  log len;
  check (len == (3 as uint));
}
