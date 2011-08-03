use std;
import std::str::*;
import std::uint::*;

fn main() {
    let a: uint = 4u;
    let b: uint = 1u;
    check (le(b, a));
    b <-> a;
    log safe_slice("kitties", a, b);
}