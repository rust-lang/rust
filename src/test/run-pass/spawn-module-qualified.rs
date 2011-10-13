use std;
import std::task::join;
import std::task::spawn_joinable2;

fn main() { let x = spawn_joinable2(10, m::child); join(x); }

mod m {
    fn# child(&&i: int) { log i; }
}
