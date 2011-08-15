use std;
import std::vec;
import std::task;
import std::uint;
import std::str;

fn f(n: uint) {
    let i = 0u;
    while i < n {
        task::join_id(task::_spawn(bind g()));
        i += 1u;
    }
}

fn g() {}

fn main(args: [str]) {

    let n = if vec::len(args) < 2u {
        10u
    } else {
        uint::parse_buf(str::bytes(args.(1)), 10u)
    };
    let i = 0u;
    while i < n {
        task::_spawn(bind f(n));
        i += 1u;
    }
}
