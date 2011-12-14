// xfail-win32 leaks
use std;
import task;
import comm;
import uint;

fn die(&&_i: ()) {
    fail;
}

fn iloop(&&_i: ()) {
    task::unsupervise();
    task::spawn((), die);
}

fn main() {
    uint::range(0u, 100u) {|_i|
        task::spawn((), iloop);
    }
}