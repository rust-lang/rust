// build-pass (FIXME(62277): could be check-pass?)

#![deny(dead_code)]

const LEN: usize = 4;

#[derive(Debug)]
struct Wrapper([u8; LEN]);

fn main() {
    println!("{:?}", Wrapper([0, 1, 2, 3]));
}
