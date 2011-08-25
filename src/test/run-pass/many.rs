// -*- rust -*-

use std;
import std::task;
import std::comm;

fn sub(parent: comm::chan<int>, id: int) {
    if id == 0 {
        comm::send(parent, 0);
    } else {
        let p = comm::port();
        let child = task::spawn(bind sub(comm::chan(p), id - 1));
        let y = comm::recv(p);
        comm::send(parent, y + 1);
    }
}

fn main() {
    let p = comm::port();
    let child = task::spawn(bind sub(comm::chan(p), 200));
    let y = comm::recv(p);
    log "transmission complete";
    log y;
    assert (y == 200);
}
