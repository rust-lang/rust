use std;
import std::task;
import std::comm;

fn main() { test00(); }

fn test00_start(c: comm::_chan[int], start: int, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { comm::send(c, start + i); i += 1; }
}

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::mk_port();
    let number_of_messages: int = 10;

    let t0 =
        task::_spawn(bind test00_start(p.mk_chan(), number_of_messages * 0,
                                       number_of_messages));
    let t1 =
        task::_spawn(bind test00_start(p.mk_chan(), number_of_messages * 1,
                                       number_of_messages));
    let t2 =
        task::_spawn(bind test00_start(p.mk_chan(), number_of_messages * 2,
                                       number_of_messages));
    let t3 =
        task::_spawn(bind test00_start(p.mk_chan(), number_of_messages * 3,
                                       number_of_messages));

    let i: int = 0;
    while i < number_of_messages {
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        i += 1;
    }

    task::join_id(t0);
    task::join_id(t1);
    task::join_id(t2);
    task::join_id(t3);

    assert (sum == number_of_messages * 4 * (number_of_messages * 4 - 1) / 2);
}