use std;
import comm;
import task;

fn main() {
    let p = comm::Port();
    let c = comm::Chan(p);
    comm::send(c, ~100);
    let v = comm::recv(p);
    assert v == ~100;
}