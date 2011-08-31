use std;
import std::istr::*;
import std::uint::*;

fn main() {
    let a: uint = 1u;
    let b: uint = 4u;
    let c: uint = 17u;
    check (le(a, b));
    c <- a;
    log safe_slice(~"kitties", c, b);
}
