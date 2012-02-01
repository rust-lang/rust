// -*- rust -*-
// xfail-test
// error-pattern:option
use std;
import vec::*;

fn main() {
    let y;
    let x : char = last(y);
}
