// error-pattern:meep
use std;
use comm::Chan;
use comm::Port;
use comm::send;
use comm::recv;

fn echo<T: Send>(c: Chan<T>, oc: Chan<Chan<T>>) {
    // Tests that the type argument in port gets
    // visited
    let p = Port::<T>();
    send(oc, Chan(p));

    let x = recv(p);
    send(c, x);
}

fn main() { fail ~"meep"; }
