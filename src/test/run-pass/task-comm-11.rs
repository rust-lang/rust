use std;

fn start(c: pipes::Chan<pipes::Chan<int>>) {
    let (ch, p) = pipes::stream();
    c.send(ch);
}

fn main() {
    let (ch, p) = pipes::stream();
    let child = task::spawn(|| start(ch) );
    let c = p.recv();
}
