use std;

import std::_int;
import std::_str::eq;

fn test_to_str() {
  assert (eq(_int::to_str(0, 10u), "0"));
  assert (eq(_int::to_str(1, 10u), "1"));
  assert (eq(_int::to_str(-1, 10u), "-1"));
  assert (eq(_int::to_str(255, 16u), "ff"));
  assert (eq(_int::to_str(100, 10u), "100"));
}

fn test_pow() {
  assert (_int::pow(0, 0u) == 1);
  assert (_int::pow(0, 1u) == 0);
  assert (_int::pow(0, 2u) == 0);
  assert (_int::pow(-1, 0u) == -1);
  assert (_int::pow(1, 0u) == 1);
  assert (_int::pow(-3, 2u) == 9);
  assert (_int::pow(-3, 3u) == -27);
  assert (_int::pow(4, 9u) == 262144);
}

fn main() {
  test_to_str();
}
