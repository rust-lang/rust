// error-pattern: reference is not valid outside of its lifetime
use std;
import std::sync;

fn main() {
    let m = ~sync::mutex();
    let mut cond = None;
    do m.lock_cond |c| {
        cond = Some(c);
    }   
    option::unwrap(cond).signal();
}
