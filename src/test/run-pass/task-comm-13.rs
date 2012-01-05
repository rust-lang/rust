use std;
import task;
import comm;
import comm::send;

fn start(c: comm::chan<int>, start: int, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { send(c, start + i); i += 1; }
}

fn main() {
    #debug("Check that we don't deadlock.");
    let p = comm::port::<int>();
    let ch = comm::chan(p);
    let a = task::spawn_joinable {|| start(ch, 0, 10); };
    task::join(a);
    #debug("Joined task");
}
