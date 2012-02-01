use std;
import str::*;
import uint::*;

fn main() unsafe {
    let a: uint = 4u;
    let b: uint = 1u;
    check (le(b, a));
    b <-> a;
    log(debug, str::unsafe::safe_slice("kitties", a, b));
}
