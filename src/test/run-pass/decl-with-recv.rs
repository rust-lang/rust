// -*- rust -*-

use std;
import comm::port;
import comm::chan;
import comm::send;
import comm::recv;

fn main() {
    let po = port();
    let ch = chan(po);
    send(ch, 10);
    let i = recv(po);
    assert (i == 10);
    send(ch, 11);
    let j = recv(po);
    assert (j == 11);
}
