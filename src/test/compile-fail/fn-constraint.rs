// error-pattern:precondition constraint (for example, uint::le(a, b)
use std;
import str::*;

fn main() unsafe {
    let a: uint = 4u;
    let b: uint = 1u;
    log(error, str::unsafe::safe_slice("kitties", a, b));
}
