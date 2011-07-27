// xfail-stage0
use std;
import std::str::*;
import std::uint::*;

fn main() {
    let a: uint = 1u;
    let b: uint = 4u;
    check (le(a, b));
    let c <- a;
    log safe_slice("kitties", c, b);
}