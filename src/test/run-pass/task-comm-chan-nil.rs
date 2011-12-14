// -*- rust -*-

use std;
import comm;

// rustboot can't transmit nils across channels because they don't have
// any size, but rustc currently can because they do have size. Whether
// or not this is desirable I don't know, but here's a regression test.
fn main() {
    let po = comm::port();
    let ch = comm::chan(po);
    comm::send(ch, ());
    let n: () = comm::recv(po);
    assert (n == ());
}
