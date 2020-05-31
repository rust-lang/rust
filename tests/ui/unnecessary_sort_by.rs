// run-rustfix

use std::cmp::Reverse;

fn id(x: isize) -> isize {
    x
}

fn main() {
    let mut vec: Vec<isize> = vec![3, 6, 1, 2, 5];
    // Forward examples
    vec.sort_by(|a, b| a.cmp(b));
    vec.sort_unstable_by(|a, b| a.cmp(b));
    vec.sort_by(|a, b| (a + 5).abs().cmp(&(b + 5).abs()));
    vec.sort_unstable_by(|a, b| id(-a).cmp(&id(-b)));
    // Reverse examples
    vec.sort_by(|a, b| b.cmp(a));
    vec.sort_by(|a, b| (b + 5).abs().cmp(&(a + 5).abs()));
    vec.sort_unstable_by(|a, b| id(-b).cmp(&id(-a)));
    // Negative examples (shouldn't be changed)
    let c = &7;
    vec.sort_by(|a, b| (b - a).cmp(&(a - b)));
    vec.sort_by(|_, b| b.cmp(&5));
    vec.sort_by(|_, b| b.cmp(c));
    vec.sort_unstable_by(|a, _| a.cmp(c));
}
