// error-pattern: cannot infer an appropriate lifetime
use std;
use std::sync;
fn main() {
    let x = ~sync::RWlock();
    let mut y = None;
    do x.write_downgrade |write_mode| {
        y = Some(x.downgrade(write_mode));
    }
    // Adding this line causes a method unification failure instead
    // do (&option::unwrap(y)).read { }
}
