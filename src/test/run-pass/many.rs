// -*- rust -*-

use std;
import task;
import comm;

fn sub(&&args: (comm::chan<int>, int)) {
    let (parent, id) = args;
    if id == 0 {
        comm::send(parent, 0);
    } else {
        let p = comm::port();
        let child = task::spawn((comm::chan(p), id - 1), sub);
        let y = comm::recv(p);
        comm::send(parent, y + 1);
    }
}

fn main() {
    let p = comm::port();
    let child = task::spawn((comm::chan(p), 200), sub);
    let y = comm::recv(p);
    log "transmission complete";
    log y;
    assert (y == 200);
}
