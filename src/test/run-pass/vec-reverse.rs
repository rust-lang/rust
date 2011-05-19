use std;
import std::vec;

fn main() {
  let vec[int] v = [10, 20];

  assert v.(0) == 10;
  assert v.(1) == 20;

  vec::reverse[int](v);

  assert v.(0) == 20;
  assert v.(1) == 10;

  auto v2 = vec::reversed[int](v);

  assert v2.(0) == 10;
  assert v2.(1) == 20;

  v.(0) = 30;

  assert v2.(0) == 10;

  // Make sure they work with 0-length vectors too.
  let vec[int] v3 = [];
  auto v4 = vec::reversed[int](v3);

  vec::reverse[int](v3);
}
