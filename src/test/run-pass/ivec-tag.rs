fn producer(c: core::oldcomm::Chan<~[u8]>) {
    core::oldcomm::send(c,
         ~[1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8,
          13u8]);
}

fn main() {
    let p: core::oldcomm::Port<~[u8]> = core::oldcomm::Port();
    let ch = core::oldcomm::Chan(&p);
    let prod = task::spawn(|| producer(ch) );

    let data: ~[u8] = core::oldcomm::recv(p);
}
