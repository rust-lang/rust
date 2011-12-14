// error-pattern:precondition constraint (for example, uint::le(a, b)
use std;
import str::*;

fn main() {
    let a: uint = 4u;
    let b: uint = 1u;
    log_err safe_slice("kitties", a, b);
}
