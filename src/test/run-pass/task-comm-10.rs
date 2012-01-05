use std;
import task;
import comm;

fn start(c: comm::chan<comm::chan<str>>) {
    let p = comm::port();
    comm::send(c, comm::chan(p));

    let a;
    let b;
    a = comm::recv(p);
    assert a == "A";
    log(error, a);
    b = comm::recv(p);
    assert b == "B";
    log(error, b);
}

fn main() {
    let p = comm::port();
    let ch = comm::chan(p);
    let child = task::spawn {|| start(ch); };

    let c = comm::recv(p);
    comm::send(c, "A");
    comm::send(c, "B");
    task::yield();
}
