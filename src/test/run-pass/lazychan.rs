// xfail-stage0
// -*- rust -*-

fn main() {
    let p: port[int] = port();
    let c = chan(p);
    let y: int;

    spawn child(c);
    p |> y;
    log "received 1";
    log y;
    assert (y == 10);

    spawn child(c);
    p |> y;
    log "received 2";
    log y;
    assert (y == 10);
}

fn child(c: chan[int]) { c <| 10; }