use std;
import std::comm;

#[test]
fn create_port_and_chan() {
    let p = comm::mk_port::<int>();
    p.mk_chan();
}

#[test]
fn send_recv() {
    let p = comm::mk_port::<int>();
    let c = p.mk_chan();

    comm::send(c, 42);
    let v = p.recv();
    log_err v;
    assert (42 == v);
}

#[test]
fn send_recv_fn() {
    let p = comm::port::<int>();
    let c = comm::chan::<int>(p);
    comm::send(c, 42);
    assert (comm::recv(p) == 42);
}

#[test]
fn send_recv_fn_infer() {
    let p = comm::port();
    let c = comm::chan(p);
    comm::send(c, 42);
    assert (comm::recv(p) == 42);
}

#[test]
fn chan_chan() {
    let p = comm::port(), p2 = comm::port::<int>();
    let c = comm::chan(p);
    comm::send(c, comm::chan(p2));
    comm::recv(p);
}
