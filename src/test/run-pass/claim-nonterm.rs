// xfail-stage0
// xfail-stage1
// tests that the pred in a claim isn't actually eval'd
use std;
import std::str::*;
import std::uint::*;

pred fails(a: uint) -> bool { fail; }

fn main() { let b: uint = 4u; claim (fails(b)); }