use std;

import comm::chan;
import comm::send;

fn main() { test05(); }

fn test05_start(&&f: fn~(int)) {
    f(22);
}

fn test05() {
    let three = ~3;
    let fn_to_send = fn~(n: int) {
        log(error, *three + n); // will copy x into the closure
        assert(*three == 3);
    };
    task::spawn(fn~[move fn_to_send]() {
        test05_start(fn_to_send);
    });
}
