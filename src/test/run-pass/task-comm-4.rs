

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p: port[int] = port();
    let c: chan[int] = chan(p);
    c <| 1;
    c <| 2;
    c <| 3;
    c <| 4;
    p |> r;
    sum += r;
    log r;
    p |> r;
    sum += r;
    log r;
    p |> r;
    sum += r;
    log r;
    p |> r;
    sum += r;
    log r;
    c <| 5;
    c <| 6;
    c <| 7;
    c <| 8;
    p |> r;
    sum += r;
    log r;
    p |> r;
    sum += r;
    log r;
    p |> r;
    sum += r;
    log r;
    p |> r;
    sum += r;
    log r;
    assert (sum == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}