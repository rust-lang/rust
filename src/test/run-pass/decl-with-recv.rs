// -*- rust -*-

extern mod std;
use comm::Port;
use comm::Chan;
use comm::send;
use comm::recv;

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
