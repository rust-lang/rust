use std;
import comm::*;
import task::*;

fn a() {
    fn doit() {
        fn b(c: chan<chan<int>>) {
            let p = port();
            send(c, chan(p));
        }
        let p = port();
        let ch = chan(p);
        spawn {|| b(ch); };
        recv(p);
    }
    let mut i = 0;
    while i < 100 {
        doit();
        i += 1;
    }
}

fn main() {
    iter::repeat(100u) {||
        spawn {|| a(); }
    }
}
