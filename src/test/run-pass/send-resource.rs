import task::*;
import comm::*;

resource test(_f: int) {
    // Do nothing
}

fn main() {
    let p = port();
    let c = chan(p);

    spawn() {||
        let p = port();
        c.send(chan(p));

        let _r = p.recv();
    }

    p.recv().send(test(42));
}