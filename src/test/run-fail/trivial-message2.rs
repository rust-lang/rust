// xfail-stage0
/*
  This program should hang on the r <- po line.
 */

fn main() {
    let port[int] po = port();
    let chan[int] ch = chan(po);

    auto r <- po;

    ch <| 42;

    log_err r;
}
