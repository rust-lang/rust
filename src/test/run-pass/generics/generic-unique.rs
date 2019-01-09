// run-pass
#![allow(dead_code)]
#![feature(box_syntax)]

struct Triple<T> { x: T, y: T, z: T }

fn box_it<T>(x: Triple<T>) -> Box<Triple<T>> { return box x; }

pub fn main() {
    let x: Box<Triple<isize>> = box_it::<isize>(Triple{x: 1, y: 2, z: 3});
    assert_eq!(x.y, 2);
}
