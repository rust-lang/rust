// -*- rust -*-

use std;
import task;
import comm;

fn sub(parent: comm::chan<int>, id: int) {
    if id == 0 {
        comm::send(parent, 0);
    } else {
        let p = comm::port();
        let ch = comm::chan(p);
        let child = task::spawn {|| sub(ch, id - 1); };
        let y = comm::recv(p);
        comm::send(parent, y + 1);
    }
}

fn main() {
    let p = comm::port();
    let ch = comm::chan(p);
    let child = task::spawn {|| sub(ch, 200); };
    let y = comm::recv(p);
    #debug("transmission complete");
    log(debug, y);
    assert (y == 200);
}
