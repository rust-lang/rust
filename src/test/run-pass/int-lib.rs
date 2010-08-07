use std;

import std._int;

fn test_to_str() {
  check (_int.to_str(0, 10u) == "0");
  check (_int.to_str(1, 10u) == "1");
  check (_int.to_str(-1, 10u) == "-1");
  check (_int.to_str(255, 16u) == "ff");
  check (_int.to_str(-71, 36u) == "-1z");
}

fn main() {
  test_to_str();
}
