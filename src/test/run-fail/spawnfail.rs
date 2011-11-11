// xfail-win32
// error-pattern:explicit
use std;
import std::task;

// We don't want to see any invalid reads
fn main() {
    fn f(&&_i: ()) {
        fail;
    }
    task::spawn((), f);
}