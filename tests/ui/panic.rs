#![warn(clippy::panic)]
#![allow(clippy::assertions_on_constants)]

fn panic() {
    let a = 2;
    panic!();
    let b = a + 2;
}

fn main() {
    panic();
}
