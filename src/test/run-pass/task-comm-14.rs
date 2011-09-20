use std;
import std::comm;
import std::task;

fn main() {
    let po = comm::port();

    // Spawn 10 tasks each sending us back one int.
    let i = 10;
    while (i > 0) {
        log i;
        task::spawn(bind child(i, comm::chan(po)));
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    let value = 0;
    while (i > 0) {
        log i;
        value = comm::recv(po);
        i = i - 1;
    }

    log "main thread exiting";
}

fn child(x: int, ch: comm::chan<int>) {
    log x;
    comm::send(ch, x);
}
