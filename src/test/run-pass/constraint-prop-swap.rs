use std;
import str::*;
import uint::*;

fn main() unsafe {
    let a: uint = 4u;
    let b: uint = 1u;
    check (le(b, a));
    b <-> a;
    log(debug, str::unsafe::slice_bytes_safe_range("kitties", a, b));
}
