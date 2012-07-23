// xfail-win32
use std;
import task;

fn f() {
    let a = ~0;
    fail;
}

fn main() {
    task::spawn_unlinked(f);
}
