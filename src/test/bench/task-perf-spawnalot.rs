use std;
import std::vec;
import std::task;
import std::uint;
import std::str;

fn f(n: uint) {
    let i = 0u;
    while i < n {
        task::join(spawn g());
        i += 1u;
    }
}

fn g() {}

fn main(args: vec[str]) {

    let n = if vec::len(args) < 2u {
        10u
    } else {
        uint::parse_buf(str::bytes(args.(1)), 10u)
    };
    let i = 0u;
    while i < n {
        spawn f(n);
        i += 1u;
    }
}