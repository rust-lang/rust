use std;
import std::task;

fn start(c: chan[chan[str]]) {
    let p: port[str];

    let a;
    let b;
    p = port();
    c <| chan(p);
    p |> a;
    log_err a;
    p |> b;
    log_err b;
}

fn main() {
    let p: port[chan[str]];
    let child;

    p = port();
    child = spawn start(chan(p));
    let c;

    p |> c;
    c <| "A";
    c <| "B";
    task::yield();
}