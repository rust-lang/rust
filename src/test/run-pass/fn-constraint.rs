use std;
import std::str::*;
import std::uint::*;

fn main() {
  let uint a = 1u;
  let uint b = 4u;
  check le(a, b);
  log (safe_slice("kitties", a, b)); 
}