// xfail-win32
use std;
import task;
import comm;

resource complainer(c: comm::chan<bool>) {
    comm::send(c, true);
}

fn f(c: comm::chan<bool>) {
    let c <- complainer(c);
    fail;
}

fn main() {
    let p = comm::port();
    let c = comm::chan(p);
    let builder = task::builder();
    task::unsupervise(builder);
    task::run(builder) {|| f(c); }
    assert comm::recv(p);
}