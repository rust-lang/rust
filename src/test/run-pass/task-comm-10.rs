use std;
import task;
import pipes;

fn start(c: pipes::chan<pipes::chan<~str>>) {
    let (ch, p) = pipes::stream();
    c.send(ch);

    let mut a;
    let mut b;
    a = p.recv();
    assert a == ~"A";
    log(error, a);
    b = p.recv();
    assert b == ~"B";
    log(error, b);
}

fn main() {
    let (ch, p) = pipes::stream();
    let child = task::spawn(|| start(ch) );

    let c = p.recv();
    c.send(~"A");
    c.send(~"B");
    task::yield();
}
