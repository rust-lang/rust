

fn main() { test00(); }

fn test00() {
    let int r = 0;
    let int sum = 0;
    let port[int] p = port();
    let chan[int] c = chan(p);
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