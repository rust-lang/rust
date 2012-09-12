// error-pattern: reference is not valid outside of its lifetime
extern mod std;
use std::sync;
fn main() {
    let x = ~sync::RWlock();
    let mut y = None;
    do x.write_downgrade |write_mode| {
        do (&write_mode).write_cond |cond| {
            y = Some(cond);
        }
    }
    option::unwrap(y).wait();
}
