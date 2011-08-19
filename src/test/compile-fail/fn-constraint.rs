// error-pattern: Unsatisfied precondition constraint (for example, le(a, b)
use std;
import std::str::*;

fn main() {
    let a: uint = 4u;
    let b: uint = 1u;
    log_err safe_slice("kitties", a, b);
}
