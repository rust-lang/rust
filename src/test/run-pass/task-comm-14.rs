use std;
import comm;
import task;

fn main() {
    let po = comm::port::<int>();

    // Spawn 10 tasks each sending us back one int.
    let i = 10;
    while (i > 0) {
        log_full(core::debug, i);
        task::spawn((i, comm::chan(po)), child);
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    let value = 0;
    while (i > 0) {
        log_full(core::debug, i);
        value = comm::recv(po);
        i = i - 1;
    }

    #debug("main thread exiting");
}

fn child(&&args: (int, comm::chan<int>)) {
    let (x, ch) = args;
    log_full(core::debug, x);
    comm::send(ch, x);
}
