// xfail-stage0
use std;
import std::str::*;
import std::uint::*;

fn main() {
  let uint a = 4u;
  let uint b = 1u;
  check le(b, a);
  b <-> a;
  log (safe_slice("kitties", a, b)); 
}