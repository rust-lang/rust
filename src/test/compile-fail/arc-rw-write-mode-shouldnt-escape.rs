// error-pattern: reference is not valid outside of its lifetime
use std;
use std::arc;
fn main() {
    let x = ~arc::RWARC(1);
    let mut y = None;
    do x.write_downgrade |write_mode| {
        y = Some(write_mode);
    }
    // Adding this line causes a method unification failure instead
    // do (&option::unwrap(y)).write |state| { assert *state == 1; }
}
