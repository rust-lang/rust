// error-pattern:meep
use std;
import comm::chan;
import comm::port;
import comm::send;
import comm::recv;

fn echo<T: send>(c: chan<T>, oc: chan<chan<T>>) {
    // Tests that the type argument in port gets
    // visited
    let p = port::<T>();
    send(oc, chan(p));

    let x = recv(p);
    send(c, x);
}

fn main() { fail "meep"; }
