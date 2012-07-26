use std;
import pipes;
import pipes::send;

fn main() { test00(); }

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let (c, p) = pipes::stream();
    c.send(1);
    c.send(2);
    c.send(3);
    c.send(4);
    r = p.recv();
    sum += r;
    log(debug, r);
    r = p.recv();
    sum += r;
    log(debug, r);
    r = p.recv();
    sum += r;
    log(debug, r);
    r = p.recv();
    sum += r;
    log(debug, r);
    c.send(5);
    c.send(6);
    c.send(7);
    c.send(8);
    r = p.recv();
    sum += r;
    log(debug, r);
    r = p.recv();
    sum += r;
    log(debug, r);
    r = p.recv();
    sum += r;
    log(debug, r);
    r = p.recv();
    sum += r;
    log(debug, r);
    assert (sum == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
