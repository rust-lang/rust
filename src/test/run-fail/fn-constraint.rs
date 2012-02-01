// error-pattern:Predicate le(a, b) failed
use std;
import str::*;
import uint::le;

fn main() unsafe {
    let a: uint = 4u;
    let b: uint = 1u;
    check (le(a, b));
    log(error, str::unsafe::safe_slice("kitties", a, b));
}
