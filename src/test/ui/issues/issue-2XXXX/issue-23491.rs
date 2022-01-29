// run-pass
#![allow(unused_variables)]
#![feature(box_syntax)]

struct Node<T: ?Sized>(T);

fn main() {
    let x: Box<Node<[isize]>> = box Node([]);
}
