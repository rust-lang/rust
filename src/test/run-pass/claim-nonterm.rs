// tests that the pred in a claim isn't actually eval'd
use std;
import std::uint::*;

pure fn fails(a: uint) -> bool { fail; }

fn main() { let b: uint = 4u; claim (fails(b)); }
