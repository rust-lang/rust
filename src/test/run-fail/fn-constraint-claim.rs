// xfail-stage0
// xfail-stage1
// error-pattern:quux
use std;
import std::str::*;
import std::uint::*;

fn nop(uint a, uint b) : le(a, b) {
  fail "quux";
}

fn main() {
  let uint a = 5u;
  let uint b = 4u;
  claim le(a, b);
  nop(a, b);
}