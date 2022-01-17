// run-pass
#![allow(unused_variables)]
#![feature(box_syntax)]

struct Node<T: ?Sized>(#[allow(dead_code)] T);

fn main() {
    let x: Box<Node<[isize]>> = box Node([]);
}
