// error-pattern: cannot infer an appropriate lifetime
use std;
import std::sync;
fn main() {
    let x = ~sync::rwlock();
    let mut y = none;
    do x.write_downgrade |write_mode| {
        y = some(x.downgrade(write_mode));
    }
    // Adding this line causes a method unification failure instead
    // do (&option::unwrap(y)).read { }
}
