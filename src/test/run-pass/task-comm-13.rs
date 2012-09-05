use std;
use pipes::send;

fn start(c: pipes::Chan<int>, start: int, number_of_messages: int) {
    let mut i: int = 0;
    while i < number_of_messages { c.send(start + i); i += 1; }
}

fn main() {
    debug!("Check that we don't deadlock.");
    let (ch, p) = pipes::stream();
    task::try(|| start(ch, 0, 10) );
    debug!("Joined task");
}
