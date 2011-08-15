// error-pattern:expected argument mode
use std;
import std::vec::map;

fn main() {
    fn f(i: uint) -> bool { true }

    let a = map(f, ~[5u]);
}