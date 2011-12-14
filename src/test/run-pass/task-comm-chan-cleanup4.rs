use std;
import int;
import comm;
import task;

// We're trying to trigger a race between send and port destruction that
// results in the string not being freed

fn starship(&&ch: comm::chan<str>) {
    int::range(0, 10) { |_i|
        comm::send(ch, "pew pew");
    }
}

fn starbase(&&_args: ()) {
    int::range(0, 10) { |_i|
        let p = comm::port();
        task::spawn(comm::chan(p), starship);
        task::yield();
    }
}

fn main() {
    int::range(0, 10) { |_i|
        task::spawn((), starbase);
    }
}