#![warn(clippy::from_iter_instead_of_collect)]

use std::collections::HashMap;
use std::iter::FromIterator;

fn main() {
    let iter_expr = std::iter::repeat(5).take(5);
    Vec::from_iter(iter_expr);

    HashMap::<usize, &i8>::from_iter(vec![5, 5, 5, 5].iter().enumerate());

    Vec::from_iter(vec![42u32]);
}
