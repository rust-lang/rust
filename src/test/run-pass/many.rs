// -*- rust -*-

use std;

fn sub(parent: comm::Chan<int>, id: int) {
    if id == 0 {
        comm::send(parent, 0);
    } else {
        let p = comm::Port();
        let ch = comm::Chan(p);
        let child = task::spawn(|| sub(ch, id - 1) );
        let y = comm::recv(p);
        comm::send(parent, y + 1);
    }
}

fn main() {
    let p = comm::Port();
    let ch = comm::Chan(p);
    let child = task::spawn(|| sub(ch, 200) );
    let y = comm::recv(p);
    debug!("transmission complete");
    log(debug, y);
    assert (y == 200);
}
