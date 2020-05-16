#![warn(clippy::sort_by_key_reverse)]

fn main() {
    let mut vec = vec![3, 6, 1, 2, 5];
    vec.sort_by(|a, b| b.cmp(a));
}
