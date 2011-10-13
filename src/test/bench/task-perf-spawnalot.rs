use std;
import std::vec;
import std::task;
import std::uint;
import std::str;

fn# f(&&n: uint) {
    let i = 0u;
    while i < n {
        task::join(task::spawn_joinable2((), g));
        i += 1u;
    }
}

fn# g(&&_i: ()) { }

fn main(args: [str]) {
    let n =
        if vec::len(args) < 2u {
            10u
        } else { uint::parse_buf(str::bytes(args[1]), 10u) };
    let i = 0u;
    while i < n { task::spawn2(copy n, f); i += 1u; }
}
