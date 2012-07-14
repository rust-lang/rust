use std;
import int;
import comm;
import task;

// We're trying to trigger a race between send and port destruction that
// results in the string not being freed

fn starship(&&ch: comm::chan<~str>) {
    for int::range(0, 10) |_i| {
        comm::send(ch, ~"pew pew");
    }
}

fn starbase() {
    for int::range(0, 10) |_i| {
        let p = comm::port();
        let c = comm::chan(p);
        task::spawn(|| starship(c) );
        task::yield();
    }
}

fn main() {
    for int::range(0, 10) |_i| {
        task::spawn(|| starbase() );
    }
}