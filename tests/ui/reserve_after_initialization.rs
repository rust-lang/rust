#![warn(clippy::reserve_after_initialization)]

fn main() {
    let mut v: Vec<usize> = vec![];
    v.reserve(10);
}
