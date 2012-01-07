use std;
import vec;
import task;
import uint;
import str;

fn f(&&n: uint) {
    let i = 0u;
    while i < n {
        task::join(task::spawn_joinable {|| g(); });
        i += 1u;
    }
}

fn g(&&_i: ()) { }

fn main(args: [str]) {
    let n =
        if vec::len(args) < 2u {
            10u
        } else { uint::parse_buf(str::bytes(args[1]), 10u) };
    let i = 0u;
    while i < n { task::spawn {|| f(n); }; i += 1u; }
}
