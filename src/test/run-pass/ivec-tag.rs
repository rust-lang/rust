extern mod std;

use comm::Chan;
use comm::Port;
use comm::send;
use comm::recv;

fn producer(c: Chan<~[u8]>) {
    send(c,
         ~[1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8,
          13u8]);
}

fn main() {
    let p: Port<~[u8]> = Port();
    let ch = Chan(p);
    let prod = task::spawn(|| producer(ch) );

    let data: ~[u8] = recv(p);
}
