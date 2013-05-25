use std::comm::*;
use std::task;

fn producer(c: &Chan<~[u8]>) {
    c.send(
         ~[1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8,
          13u8]);
}

pub fn main() {
    let (p, ch) = stream::<~[u8]>();
    let prod = task::spawn(|| producer(&ch) );

    let data: ~[u8] = p.recv();
}
