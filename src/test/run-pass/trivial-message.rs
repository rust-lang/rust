import pipes::{port, chan};

/*
  This is about the simplest program that can successfully send a
  message.
 */
fn main() {
    let (ch, po) = pipes::stream();
    ch.send(42);
    let r = po.recv();
    log(error, r);
}
