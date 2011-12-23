use std;
import comm;
import comm::send;

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::port();
    let c = comm::chan(p);
    send(c, 1);
    send(c, 2);
    send(c, 3);
    send(c, 4);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    send(c, 5);
    send(c, 6);
    send(c, 7);
    send(c, 8);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    r = comm::recv(p);
    sum += r;
    log(debug, r);
    assert (sum == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
