// xfail-stage0
// -*- rust -*-

fn main() {
    let p: port[int] = port();
    let t: task = spawn child(chan(p));
    let y: int;
    p |> y;
    log_err "received";
    log_err y;
    assert (y == 10);
}

fn child(c: chan[int]) { log_err "sending"; c <| 10; log_err "value sent" }