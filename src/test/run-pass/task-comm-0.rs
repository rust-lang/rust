// xfail-stage0
fn main() { test05(); }

fn test05_start(ch: chan[int]) { ch <| 10; ch <| 20; ch <| 30; }

fn test05() {
    let po: port[int] = port();
    let ch: chan[int] = chan(po);
    spawn test05_start(chan(po));
    let value: int;
    po |> value;
    po |> value;
    po |> value;
    assert (value == 30);
}