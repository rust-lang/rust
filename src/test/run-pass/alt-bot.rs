

// xfail-stage0
use std;
import std::option::*;

fn main() {
    let int i =
        alt (some[int](3)) {
            case (none[int]) { fail }
            case (some[int](_)) { 5 }
        };
    log i;
}