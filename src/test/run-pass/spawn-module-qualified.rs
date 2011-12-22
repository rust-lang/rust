use std;
import task::join;
import task::spawn_joinable;

fn main() { let x = spawn_joinable(10, m::child); join(x); }

mod m {
    fn child(&&i: int) { log_full(core::debug, i); }
}
