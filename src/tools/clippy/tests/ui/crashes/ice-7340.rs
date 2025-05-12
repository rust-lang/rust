//@ check-pass

#![allow(clippy::no_effect)]

fn main() {
    const CONSTANT: usize = 8;
    [1; 1 % CONSTANT];
}
