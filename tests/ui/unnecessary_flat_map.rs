#![warn(clippy::flat_map)]

fn main() {
    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    iterator.flat_map(|x| x);
}
