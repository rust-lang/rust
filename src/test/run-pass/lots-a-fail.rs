// xfail-win32 leaks
use std;
import task;
import comm;
import uint;

fn die() {
    fail;
}

fn iloop() {
    task::spawn(|| die() );
}

fn main() {
    for uint::range(0u, 100u) |_i| {
        task::spawn_unlinked(|| iloop() );
    }
}
