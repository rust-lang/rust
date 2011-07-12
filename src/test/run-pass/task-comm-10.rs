// xfail-stage0
// xfail-stage1
// xfail-stage2

use std;
import std::task;

fn start(chan[chan[str]] c) {
    let port[str] p = port();
    c <| chan(p);
    auto a; p |> a;
    // auto b; p |> b; // Never read the second string.
}

fn main() {
    let port[chan[str]] p = port();
    auto child = spawn start(chan(p));
    auto c; p |> c;
    c <| "A";
    c <| "B";
    task::yield();
}
