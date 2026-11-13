//@ run-fail
//@ error-pattern:index out of bounds
//@ needs-subprocess

use std::mem::size_of;

fn main() {
    let xs = [1, 2, 3];
    xs[usize::MAX / size_of::<isize>() + 1];
}
