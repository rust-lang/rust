use std;
import std::comm::mk_port;
import std::comm::send;

/*
  This is about the simplest program that can successfully send a
  message.
 */
fn main() {
    let po = mk_port::<int>();
    let ch = po.mk_chan();
    send(ch, 42);
    let r = po.recv();
    log_err r;
}
