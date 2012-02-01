use std;
import str::*;
import uint::*;

fn main() unsafe {
    let a: uint = 1u;
    let b: uint = 4u;
    check (le(a, b));
    let c = b;
    log(debug, str::unsafe::slice_bytes_safe_range("kitties", a, c));
}
