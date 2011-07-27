// xfail-stage0

use std;
import std::task;

fn main() { test00(); }

fn test00_start(c: chan[int], number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { c <| i; i += 1; }
}

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p: port[int] = port();
    let number_of_messages: int = 10;

    let t0: task = spawn test00_start(chan(p), number_of_messages);

    let i: int = 0;
    while i < number_of_messages { p |> r; sum += r; log r; i += 1; }

    task::join(t0);

    assert (sum == number_of_messages * (number_of_messages - 1) / 2);
}