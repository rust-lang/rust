use std;
import std::task;
import std::comm;

fn main() { test00(); }

fn# test00_start(&&args: (comm::chan<int>, int, int)) {
    let (c, start, number_of_messages) = args;
    let i: int = 0;
    while i < number_of_messages { comm::send(c, start + i); i += 1; }
}

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::port();
    let number_of_messages: int = 10;

    let t0 =
        task::spawn_joinable((comm::chan(p),
                               number_of_messages * 0,
                               number_of_messages), test00_start);
    let t1 =
        task::spawn_joinable((comm::chan(p),
                               number_of_messages * 1,
                               number_of_messages), test00_start);
    let t2 =
        task::spawn_joinable((comm::chan(p),
                               number_of_messages * 2,
                               number_of_messages), test00_start);
    let t3 =
        task::spawn_joinable((comm::chan(p),
                               number_of_messages * 3,
                               number_of_messages), test00_start);

    let i: int = 0;
    while i < number_of_messages {
        r = comm::recv(p);
        sum += r;
        r = comm::recv(p);
        sum += r;
        r = comm::recv(p);
        sum += r;
        r = comm::recv(p);
        sum += r;
        i += 1;
    }

    task::join(t0);
    task::join(t1);
    task::join(t2);
    task::join(t3);

    assert (sum == number_of_messages * 4 * (number_of_messages * 4 - 1) / 2);
}
