#![warn(clippy::type_complexity)]
use std::iter::{Filter, Map};
use std::vec::IntoIter;

struct S;

impl IntoIterator for S {
    type Item = i32;
    // Should not warn since there is no way to simplify this
    type IntoIter = Filter<Map<IntoIter<i32>, fn(i32) -> i32>, fn(&i32) -> bool>;

    fn into_iter(self) -> Self::IntoIter {
        fn m(a: i32) -> i32 {
            a
        }
        fn p(_: &i32) -> bool {
            true
        }
        vec![1i32, 2, 3].into_iter().map(m as fn(_) -> _).filter(p)
    }
}

fn main() {}
