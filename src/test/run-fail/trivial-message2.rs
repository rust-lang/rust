


// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3

/*
  This program should hang on the po |> r line.
 */
fn main() {
    let port[int] po = port();
    let chan[int] ch = chan(po);
    auto r;
    po |> r;
    ch <| 42;
    log_err r;
}