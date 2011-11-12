use std;
import std::comm::*;
import std::task::*;

fn a(&&_args: ()) {
    fn doit() {
        fn b(c: chan<chan<int>>) {
            let p = port();
            send(c, chan(p));
        }
        let p = port();
        spawn(chan(p), b);
        recv(p);
    }
    let i = 0;
    while i < 100 {
        doit();
        i += 1;
    }
}

fn main() {
    let t = spawn_joinable((), a);
    join(t);
}
