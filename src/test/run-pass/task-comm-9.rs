// xfail-stage0

use std;
import std::task;

fn main() -> () {
   test00();
}

fn test00_start(chan[int] c, int number_of_messages) {
    let int i = 0;
    while (i < number_of_messages) {
        c <| i;
        i += 1;
    }
}

fn test00() {
    let int r = 0;
    let int sum = 0;
    let port[int] p = port();
    let int number_of_messages = 10;

    let task t0 = spawn
        test00_start(chan(p), number_of_messages);

    let int i = 0;
    while (i < number_of_messages) {
        p |> r; sum += r; log (r);
        i += 1;
    }

    task::join(t0);

    assert (sum == (number_of_messages * (number_of_messages - 1)) / 2);
}