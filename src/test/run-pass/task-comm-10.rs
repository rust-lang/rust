use std;
import std::task;
import std::comm;

fn start(c: comm::chan<comm::chan<str>>) {
    let p = comm::port();
    comm::send(c, comm::chan(p));

    let a;
    let b;
    a = comm::recv(p);
    assert a == "A";
    log_err a;
    b = comm::recv(p);
    assert b == "B";
    log_err b;
}

fn main() {
    let p = comm::port();
    let child = task::spawn(comm::chan(p), start);

    let c = comm::recv(p);
    comm::send(c, "A");
    comm::send(c, "B");
    task::yield();
}
