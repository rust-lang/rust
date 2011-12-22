// error-pattern:Predicate le(a, b) failed
use std;
import str::*;
import uint::le;

fn main() {
    let a: uint = 4u;
    let b: uint = 1u;
    check (le(a, b));
    log_full(core::error, safe_slice("kitties", a, b));
}
