// xfail-stage0
// xfail-stage1
// error-pattern:quux
use std;
import std::str::*;
import std::uint::*;

fn nop(a: uint, b: uint) { fail "quux"; }

fn main() { let a: uint = 5u; let b: uint = 4u; claim (le(a, b)); nop(a, b); }