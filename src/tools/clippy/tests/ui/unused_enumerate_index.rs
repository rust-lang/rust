#![allow(unused)]
#![warn(clippy::unused_enumerate_index)]

use std::iter::Enumerate;

fn main() {
    let v = [1, 2, 3];
    for (_, x) in v.iter().enumerate() {
        println!("{x}");
    }

    struct Dummy1;
    impl Dummy1 {
        fn enumerate(self) -> Vec<usize> {
            vec![]
        }
    }
    let dummy = Dummy1;
    for x in dummy.enumerate() {
        println!("{x}");
    }

    struct Dummy2;
    impl Dummy2 {
        fn enumerate(self) -> Enumerate<std::vec::IntoIter<usize>> {
            vec![1, 2].into_iter().enumerate()
        }
    }
    let dummy = Dummy2;
    for (_, x) in dummy.enumerate() {
        println!("{x}");
    }

    let mut with_used_iterator = [1, 2, 3].into_iter().enumerate();
    with_used_iterator.next();
    for (_, x) in with_used_iterator {
        println!("{x}");
    }

    struct Dummy3(std::vec::IntoIter<usize>);

    impl Iterator for Dummy3 {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    let dummy = Dummy3(vec![1, 2, 3].into_iter());
    for (_, x) in dummy.enumerate() {
        println!("{x}");
    }
}
