#![warn(clippy::reserve_after_initialization)]

fn main() {
    // Should lint
    let mut v1: Vec<usize> = vec![];
    v1.reserve(10);

    // Should lint
    let capacity = 10;
    let mut v2: Vec<usize> = vec![];
    v2.reserve(capacity);

    // Shouldn't lint
    let mut v3 = vec![1];
    v3.reserve(10);

    // Shouldn't lint
    let mut v4: Vec<usize> = Vec::with_capacity(10);
}
