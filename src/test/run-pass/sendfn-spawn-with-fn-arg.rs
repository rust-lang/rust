use std;

import std::comm;
import std::comm::chan;
import std::comm::send;

fn main() { test05(); }

fn test05_start(&&f: sendfn(int)) {
    f(22);
}

fn test05() {
    let three = ~3;
    let fn_to_send = sendfn(n: int) {
        log_err *three + n; // will copy x into the closure
        assert(*three == 3);
    };
    task::spawn(fn_to_send, test05_start);
}
