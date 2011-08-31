// xfail-test

/*
  This program should hang on the po |> r line.
 */
fn main() {
    let po: port<int> = port();
    let ch: chan<int> = chan(po);
    let r;
    po |> r;
    ch <| 42;
    log_err r;
}
