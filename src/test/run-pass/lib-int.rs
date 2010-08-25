use std;

import std._int;
import std._str.eq;

fn test_to_str() {
  check (eq(_int.to_str(0, 10u), "0"));
  check (eq(_int.to_str(1, 10u), "1"));
  check (eq(_int.to_str(-1, 10u), "-1"));
  check (eq(_int.to_str(255, 16u), "ff"));
  check (eq(_int.to_str(100, 10u), "100"));
}

fn main() {
  test_to_str();
}
