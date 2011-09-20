// error-pattern:explicit
use std;
import std::task;

// We don't want to see any invalid reads
fn main() {
    fn f() {
        fail;
    }
    let g = f;
    task::spawn(g);
}