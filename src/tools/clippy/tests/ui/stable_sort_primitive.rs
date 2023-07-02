//@run-rustfix
#![warn(clippy::stable_sort_primitive)]
#![allow(clippy::useless_vec)]

fn main() {
    // positive examples
    let mut vec = vec![1, 3, 2];
    vec.sort();
    let mut vec = vec![false, false, true];
    vec.sort();
    let mut vec = vec!['a', 'A', 'c'];
    vec.sort();
    let mut vec = vec!["ab", "cd", "ab", "bc"];
    vec.sort();
    let mut vec = vec![(2, 1), (1, 2), (2, 5)];
    vec.sort();
    let mut vec = vec![[2, 1], [1, 2], [2, 5]];
    vec.sort();
    let mut arr = [1, 3, 2];
    arr.sort();
    // Negative examples: behavior changes if made unstable
    let mut vec = vec![1, 3, 2];
    vec.sort_by_key(|i| i / 2);
    vec.sort_by(|&a, &b| (a + b).cmp(&b));
    // negative examples - Not of a primitive type
    let mut vec_of_complex = vec![String::from("hello"), String::from("world!")];
    vec_of_complex.sort();
    vec_of_complex.sort_by_key(String::len);
    let mut vec = vec![(String::from("hello"), String::from("world"))];
    vec.sort();
    let mut vec = vec![[String::from("hello"), String::from("world")]];
    vec.sort();
}
