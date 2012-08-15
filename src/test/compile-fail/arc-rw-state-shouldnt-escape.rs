// error-pattern: reference is not valid outside of its lifetime
use std;
import std::arc;
fn main() {
    let x = ~arc::rw_arc(1);
    let mut y = none;
    do x.write |one| {
        y = some(one);
    }
    *option::unwrap(y) = 2;
}
