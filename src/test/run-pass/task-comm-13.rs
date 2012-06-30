use std;
import task;
import comm;
import comm::send;

fn start(c: comm::chan<int>, start: int, number_of_messages: int) {
    let mut i: int = 0;
    while i < number_of_messages { send(c, start + i); i += 1; }
}

fn main() {
    #debug("Check that we don't deadlock.");
    let p = comm::port::<int>();
    let ch = comm::chan(p);
    task::try(|| start(ch, 0, 10) );
    #debug("Joined task");
}
