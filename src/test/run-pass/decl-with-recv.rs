// -*- rust -*-

use std;
import comm::Port;
import comm::Chan;
import comm::send;
import comm::recv;

fn main() {
    let po = Port();
    let ch = Chan(po);
    send(ch, 10);
    let i = recv(po);
    assert (i == 10);
    send(ch, 11);
    let j = recv(po);
    assert (j == 11);
}
