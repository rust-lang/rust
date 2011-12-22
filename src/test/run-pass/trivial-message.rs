use std;
import comm::*;

/*
  This is about the simplest program that can successfully send a
  message.
 */
fn main() {
    let po = port();
    let ch = chan(po);
    send(ch, 42);
    let r = recv(po);
    log_full(core::error, r);
}
