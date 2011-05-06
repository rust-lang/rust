use std;

import std.Int;
import std.Str.eq;

fn test_to_str() {
  assert (eq(Int.to_str(0, 10u), "0"));
  assert (eq(Int.to_str(1, 10u), "1"));
  assert (eq(Int.to_str(-1, 10u), "-1"));
  assert (eq(Int.to_str(255, 16u), "ff"));
  assert (eq(Int.to_str(100, 10u), "100"));
}

fn test_pow() {
  assert (Int.pow(0, 0u) == 1);
  assert (Int.pow(0, 1u) == 0);
  assert (Int.pow(0, 2u) == 0);
  assert (Int.pow(-1, 0u) == -1);
  assert (Int.pow(1, 0u) == 1);
  assert (Int.pow(-3, 2u) == 9);
  assert (Int.pow(-3, 3u) == -27);
  assert (Int.pow(4, 9u) == 262144);
}

fn main() {
  test_to_str();
}
