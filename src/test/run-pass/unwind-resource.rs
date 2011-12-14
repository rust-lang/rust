// xfail-win32
use std;
import task;
import comm;

resource complainer(c: comm::chan<bool>) {
    comm::send(c, true);
}

fn f(c: comm::chan<bool>) {
    task::unsupervise();
    let c <- complainer(c);
    fail;
}

fn main() {
    let p = comm::port();
    let c = comm::chan(p);
    task::spawn(c, f);
    assert comm::recv(p);
}