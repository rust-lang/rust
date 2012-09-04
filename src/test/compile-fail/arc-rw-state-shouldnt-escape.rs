// error-pattern: reference is not valid outside of its lifetime
use std;
import std::arc;
fn main() {
    let x = ~arc::RWARC(1);
    let mut y = None;
    do x.write |one| {
        y = Some(one);
    }
    *option::unwrap(y) = 2;
}
