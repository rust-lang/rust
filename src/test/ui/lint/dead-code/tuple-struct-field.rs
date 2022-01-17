#![deny(dead_code)]

const LEN: usize = 4;

#[derive(Debug)]
struct Wrapper([u8; LEN]); //~ ERROR field is never read

fn main() {
    println!("{:?}", Wrapper([0, 1, 2, 3]));
}
