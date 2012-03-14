// error-pattern:precondition constraint (for example, uint::le(a, b)
use std;
import str::*;

fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let a: uint = 4u;
    let b: uint = 1u;
    log(error, foo(a, b));
}
