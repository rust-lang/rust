use std;
import std::comm;

#[test]
fn create_port_and_chan() {
    let p = comm::mk_port[int]();
    let c = p.mk_chan();
}
