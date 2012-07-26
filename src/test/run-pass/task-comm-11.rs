use std;
import pipes;
import task;

fn start(c: pipes::chan<pipes::chan<int>>) {
    let (ch, p) = pipes::stream();
    c.send(ch);
}

fn main() {
    let (ch, p) = pipes::stream();
    let child = task::spawn(|| start(ch) );
    let c = p.recv();
}
