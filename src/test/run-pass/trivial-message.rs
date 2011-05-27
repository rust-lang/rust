/*
  This is about the simplest program that can successfully send a
  message.
 */

fn main() {
    let port[int] po = port();
    let chan[int] ch = chan(po);

    ch <| 42;

    auto r; r <- po;

    log_err r;
}
