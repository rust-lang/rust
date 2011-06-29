// xfail-stage0
// xfail-stage1
// tests that the pred in a claim isn't actually eval'd
use std;
import std::str::*;
import std::uint::*;

pred fails(uint a) -> bool {
  fail;
}

fn main() {
  let uint a = 5u;
  let uint b = 4u;
  claim fails(b);
}