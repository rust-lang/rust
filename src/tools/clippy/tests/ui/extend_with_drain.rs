#![warn(clippy::extend_with_drain)]
#![allow(clippy::iter_with_drain)]
use std::collections::BinaryHeap;
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

    //won't get linted because it is not a vec

    let mut heap = BinaryHeap::from(vec![1, 3]);
    let mut heap2 = BinaryHeap::from(vec![]);
    heap2.extend(heap.drain());

    let mut x = vec![0, 1, 2, 3, 5];
    let ref_x = &mut x;
    let mut y = Vec::new();
    y.extend(ref_x.drain(..));
    //~^ extend_with_drain
}

fn return_vector() -> Vec<u8> {
    let mut new_vector = vec![];

    for i in 1..10 {
        new_vector.push(i)
    }

    new_vector
}
