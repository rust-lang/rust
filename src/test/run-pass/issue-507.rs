
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

extern mod std;

use comm::Chan;
use comm::send;
use comm::Port;
use comm::recv;

fn grandchild(c: Chan<int>) { send(c, 42); }

fn child(c: Chan<int>) {
    task::spawn(|| grandchild(c) )
}

fn main() {
    let p = comm::Port();
    let ch = Chan(p);

    task::spawn(|| child(ch) );

    let x: int = recv(p);

    log(debug, x);

    assert (x == 42);
}
