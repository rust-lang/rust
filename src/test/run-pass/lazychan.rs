// -*- rust -*-

fn main() {
    let p: port[int] = port();
    let y: int;

    spawn child(chan(p));
    p |> y;
    log "received 1";
    log y;
    assert (y == 10);

    spawn child(chan(p));
    p |> y;
    log "received 2";
    log y;
    assert (y == 10);
}

fn child(c: chan[int]) { c <| 10; }