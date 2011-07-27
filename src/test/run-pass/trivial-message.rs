


/*
  This is about the simplest program that can successfully send a
  message.
 */
fn main() {
    let po: port[int] = port();
    let ch: chan[int] = chan(po);
    ch <| 42;
    let r;
    po |> r;
    log_err r;
}