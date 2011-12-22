// xfail-test
// -*- rust -*-

use std;
import comm;
import task;

fn f(c: comm::_chan<int>) {
    type t = {_0: int, _1: int, _2: int};

    // Allocate a box.
    let x: @t = @{_0: 1, _1: 2, _2: 3};

    // Signal parent that we've allocated a box.
    comm::send(c, 1);


    while true {
        // spin waiting for the parent to kill us.
        #debug("child waiting to die...");

        // while waiting to die, the messages we are
        // sending to the channel are never received
        // by the parent, therefore this test cases drops
        // messages on the floor
        comm::send(c, 1);
    }
}

fn main() {
    let p = comm::mk_port();
    task::_spawn(bind f(p.mk_chan()));
    let i: int;

    // synchronize on event from child.
    i = p.recv();

    #debug("parent exiting, killing child");
}
