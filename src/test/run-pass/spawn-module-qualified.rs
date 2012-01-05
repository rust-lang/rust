use std;
import task::join;
import task::spawn_joinable;

fn main() { let x = spawn_joinable {|| m::child(10); }; join(x); }

mod m {
    fn child(&&i: int) { log(debug, i); }
}
