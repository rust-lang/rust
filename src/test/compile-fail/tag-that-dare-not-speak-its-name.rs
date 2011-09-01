// -*- rust -*-
// xfail-test
// error-pattern:mismatch
use std;
import std::vec::*;

fn main() {
    let y;
    let x : char = last(y);
}
