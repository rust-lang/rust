// error-pattern: reference is not valid outside of its lifetime
use std;
import std::sync;
fn main() {
    let x = ~sync::rwlock();
    let mut y = None;
    do x.write_cond |cond| {
        y = Some(cond);
    }
    option::unwrap(y).wait();
}
