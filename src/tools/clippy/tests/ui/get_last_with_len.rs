//@run-rustfix

#![warn(clippy::get_last_with_len)]
#![allow(unused, clippy::useless_vec)]

use std::collections::VecDeque;

fn dont_use_last() {
    let x = vec![2, 3, 5];
    let _ = x.get(x.len() - 1);
}

fn indexing_two_from_end() {
    let x = vec![2, 3, 5];
    let _ = x.get(x.len() - 2);
}

fn index_into_last() {
    let x = vec![2, 3, 5];
    let _ = x[x.len() - 1];
}

fn use_last_with_different_vec_length() {
    let x = vec![2, 3, 5];
    let y = vec!['a', 'b', 'c'];
    let _ = x.get(y.len() - 1);
}

struct S {
    field: Vec<usize>,
}

fn in_field(s: &S) {
    let _ = s.field.get(s.field.len() - 1);
}

fn main() {
    let slice = &[1, 2, 3];
    let _ = slice.get(slice.len() - 1);

    let array = [4, 5, 6];
    let _ = array.get(array.len() - 1);

    let deq = VecDeque::from([7, 8, 9]);
    let _ = deq.get(deq.len() - 1);

    let nested = [[1]];
    let _ = nested[0].get(nested[0].len() - 1);
}
