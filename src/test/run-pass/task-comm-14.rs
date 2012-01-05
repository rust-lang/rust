use std;
import comm;
import task;

fn main() {
    let po = comm::port::<int>();
    let ch = comm::chan(po);

    // Spawn 10 tasks each sending us back one int.
    let i = 10;
    while (i > 0) {
        log(debug, i);
        task::spawn {|| child(i, ch); };
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    let value = 0;
    while (i > 0) {
        log(debug, i);
        value = comm::recv(po);
        i = i - 1;
    }

    #debug("main thread exiting");
}

fn child(x: int, ch: comm::chan<int>) {
    log(debug, x);
    comm::send(ch, copy x);
}
