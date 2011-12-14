// -*- rust -*-
// xfail-test
// error-pattern:option::t
use std;
import vec::*;

fn main() {
    let y;
    let x : char = last(y);
}
