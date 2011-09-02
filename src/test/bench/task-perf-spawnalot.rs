use std;
import std::vec;
import std::task;
import std::uint;
import std::str;

fn f(n: uint) {
    let i = 0u;
    while i < n {
        let thunk = g;
        task::join(task::spawn_joinable(thunk)); i += 1u; }
}

fn g() { }

fn main(args: [istr]) {
    let n =
        if vec::len(args) < 2u {
            10u
        } else { uint::parse_buf(str::bytes(args[1]), 10u) };
    let i = 0u;
    while i < n { task::spawn(bind f(n)); i += 1u; }
}
