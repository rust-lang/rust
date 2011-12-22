use std;
import str::*;
import uint::*;

fn main() {
    let a: uint = 1u;
    let b: uint = 4u;
    let c: uint = 17u;
    check (le(a, b));
    c <- a;
    log_full(core::debug, safe_slice("kitties", c, b));
}
