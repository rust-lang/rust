// xfail-win32
use std;

fn f() {
    let a = @0;
    fail;
}

fn main() {
    task::spawn_unlinked(f);
}
