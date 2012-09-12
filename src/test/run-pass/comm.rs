// -*- rust -*-

extern mod std;
use comm::Chan;
use comm::send;
use comm::recv;

fn main() {
    let p = comm::Port();
    let ch = comm::Chan(p);
    let t = task::spawn(|| child(ch) );
    let y = recv(p);
    error!("received");
    log(error, y);
    assert (y == 10);
}

fn child(c: Chan<int>) {
    error!("sending");
    send(c, 10);
    error!("value sent");
}
