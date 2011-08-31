// error-pattern:Predicate le(a, b) failed
use std;
import std::istr::*;
import std::uint::le;

fn main() {
    let a: uint = 4u;
    let b: uint = 1u;
    check (le(a, b));
    log_err safe_slice(~"kitties", a, b);
}
