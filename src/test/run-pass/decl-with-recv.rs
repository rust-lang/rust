


// -*- rust -*-
fn main() {
    let po: port[int] = port();
    let ch: chan[int] = chan(po);
    ch <| 10;
    let i: int;
    po |> i;
    assert (i == 10);
    ch <| 11;
    let j;
    po |> j;
    assert (j == 11);
}