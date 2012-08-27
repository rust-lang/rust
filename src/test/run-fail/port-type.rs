// error-pattern:meep
use std;
import comm::Chan;
import comm::Port;
import comm::send;
import comm::recv;

fn echo<T: send>(c: Chan<T>, oc: Chan<Chan<T>>) {
    // Tests that the type argument in port gets
    // visited
    let p = Port::<T>();
    send(oc, Chan(p));

    let x = recv(p);
    send(c, x);
}

fn main() { fail ~"meep"; }
