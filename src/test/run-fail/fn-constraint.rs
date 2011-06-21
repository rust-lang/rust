// error-pattern:Predicate le(a, b) failed
use std;
import std::str::*;
import std::uint::le;

fn main() {
  let uint a = 4u;
  let uint b = 1u;
  check le(a, b);
  log_err (safe_slice("kitties", a, b));
}