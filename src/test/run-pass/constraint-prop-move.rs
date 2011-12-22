use std;
import str::*;
import uint::*;

fn main() {
    let a: uint = 1u;
    let b: uint = 4u;
    check (le(a, b));
    let c <- a;
    log_full(core::debug, safe_slice("kitties", c, b));
}
