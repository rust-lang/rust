use std;
import std::option::*;

pure fn p(x: int) -> bool { true }

fn f(x: int) : p(x) { }

fn main() {
    alt some(5) { some(y) { check (p(y)); f(y); } _ { fail "yuck"; } }
}
