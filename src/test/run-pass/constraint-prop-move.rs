// xfail-stage0
use std;
import std::str::*;
import std::uint::*;

fn main() {
  let uint a = 1u;
  let uint b = 4u;
  check le(a, b);
  auto c <- a;
  log (safe_slice("kitties", c, b));
}