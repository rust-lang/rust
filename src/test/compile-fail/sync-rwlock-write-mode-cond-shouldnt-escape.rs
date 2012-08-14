// error-pattern: reference is not valid outside of its lifetime
use std;
import std::sync;
fn main() {
    let x = ~sync::rwlock();
    let mut y = none;
    do x.write_downgrade |write_mode| {
        do (&write_mode).write_cond |cond| {
            y = some(cond);
        }
    }
    option::unwrap(y).wait();
}
