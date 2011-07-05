// xfail-stage0
// error-pattern:moop
use std;
import std::uint;
fn main() {
  for each (uint i in uint::range(0u, 10u)) {
    fail "moop";
  }
}
