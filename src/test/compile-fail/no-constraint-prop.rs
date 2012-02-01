// error-pattern:Unsatisfied precondition constraint (for example, le(b, d
use std;
import str::*;
import uint::*;

fn main() unsafe {
    let a: uint = 1u;
    let b: uint = 4u;
    let c: uint = 5u;
    // make sure that the constraint le(b, a) exists...
    check (le(b, a));
    // ...invalidate it...
    b += 1u;
    check (le(c, a));
    // ...and check that it doesn't get set in the poststate of
    // the next statement, since it's not true in the
    // prestate.
    let d <- a;
    log(debug, str::unsafe::slice_bytes_safe_range("kitties", b, d));
}
