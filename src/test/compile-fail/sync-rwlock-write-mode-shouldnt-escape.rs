// error-pattern: reference is not valid outside of its lifetime
extern mod std;
use std::sync;
fn main() {
    let x = ~sync::RWlock();
    let mut y = None;
    do x.write_downgrade |write_mode| {
        y = Some(write_mode);
    }
    // Adding this line causes a method unification failure instead
    // do (&option::unwrap(y)).write { }
}
