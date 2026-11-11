#![warn(clippy::extend_with_drain)]

use std::collections::{BinaryHeap, VecDeque};

struct NotAStdCollection {}
impl NotAStdCollection {
    /// The lint should *not* use this method because it doesn't know what it does,
    /// despite looking similar.
    fn append(&mut self, _other: &mut Self) {
        unimplemented!()
    }
    fn drain<R: core::ops::RangeBounds<usize>>(&mut self, _range: R) -> impl Iterator<Item = ()> {
        core::iter::empty()
    }
}
impl Extend<()> for NotAStdCollection {
    fn extend<T: IntoIterator<Item = ()>>(&mut self, _iter: T) {
        unimplemented!()
    }
}

fn main() {
    //gets linted
    let mut vec1 = vec![0u8; 1024];
    let mut vec2: std::vec::Vec<u8> = Vec::new();
    vec2.extend(vec1.drain(..));
    //~^ extend_with_drain

    let mut vec3 = vec![0u8; 1024];
    let mut vec4: std::vec::Vec<u8> = Vec::new();

    vec4.extend(vec3.drain(..));
    //~^ extend_with_drain

    let mut vec11: std::vec::Vec<u8> = Vec::new();

    vec11.extend(return_vector().drain(..));
    //~^ extend_with_drain

    //won't get linted it doesn't move the entire content of a vec into another
    let mut test1 = vec![0u8, 10];
    let mut test2: std::vec::Vec<u8> = Vec::new();

    test2.extend(test1.drain(4..10));

    let mut vec3 = vec![0u8; 104];
    let mut vec7: std::vec::Vec<u8> = Vec::new();

    vec3.append(&mut vec7);

    let mut vec5 = vec![0u8; 1024];
    let mut vec6: std::vec::Vec<u8> = Vec::new();

    vec5.extend(vec6.drain(..4));

    let mut vec9: std::vec::Vec<u8> = Vec::new();

    return_vector().append(&mut vec9);

    let mut x = vec![0, 1, 2, 3, 5];
    let ref_x = &mut x;
    let mut y = Vec::new();
    y.extend(ref_x.drain(..));
    //~^ extend_with_drain

    // VecDeque works the same as Vec and gets the same lint
    let mut vec_deque1 = VecDeque::from([0u8; 1024]);
    let mut vec_deque2 = VecDeque::new();
    vec_deque2.extend(vec_deque1.drain(..));
    //~^ extend_with_drain

    // BinaryHeap is the same except for not having a range parameter
    let mut heap = BinaryHeap::from(vec![1, 3]);
    let mut heap2 = BinaryHeap::from(vec![]);
    heap2.extend(heap.drain());
    //~^ extend_with_drain

    // Unknown types should not get the same treatment
    let mut foreign1 = NotAStdCollection {};
    let mut foreign2 = NotAStdCollection {};
    foreign2.extend(foreign1.drain(..));
}

fn return_vector() -> Vec<u8> {
    let mut new_vector = vec![];

    for i in 1..10 {
        new_vector.push(i)
    }

    new_vector
}
