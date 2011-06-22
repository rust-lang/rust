// xfail-stage0
fn main() -> () {
    test05();
}

fn test05_start(chan[int] ch) {
    ch <| 10;
    ch <| 20;
    ch <| 30;
}

fn test05() {
    let port[int] po = port();
    let chan[int] ch = chan(po);
    spawn test05_start(chan(po));
    let int value; po |> value;
    po |> value;
    po |> value;
    assert (value == 30);
}
