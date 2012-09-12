// error-pattern: reference is not valid outside of its lifetime
extern mod std;
use std::arc;
fn main() {
    let x = ~arc::RWARC(1);
    let mut y = None;
    do x.write_cond |_one, cond| {
        y = Some(cond);
    }
    option::unwrap(y).wait();
}
