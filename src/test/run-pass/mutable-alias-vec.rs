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
  auto len = std::_vec::len[int](v);
  log len;
  assert (len == (3 as uint));
}
