// error-pattern:expected argument mode
use std;
import std::vec::map;

fn main() {
  fn f(uint i) -> bool { true }

  auto a = map(f, [5u]);
}