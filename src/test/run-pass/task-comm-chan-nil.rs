// -*- rust -*-

extern mod std;

// rustboot can't transmit nils across channels because they don't have
// any size, but rustc currently can because they do have size. Whether
// or not this is desirable I don't know, but here's a regression test.
fn main() {
    let po = comm::Port();
    let ch = comm::Chan(po);
    comm::send(ch, ());
    let n: () = comm::recv(po);
    assert (n == ());
}
