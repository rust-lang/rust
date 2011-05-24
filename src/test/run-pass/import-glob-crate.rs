
use std;
import std::vec::*;

fn main() {
  auto v = empty[int]();
  v += [4,2];
  assert(reversed(v) == [2,4]);
}