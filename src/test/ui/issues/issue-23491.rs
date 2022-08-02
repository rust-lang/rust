// run-pass
#![allow(unused_variables)]
#![feature(box_syntax)]

struct Node<T: ?Sized>(#[allow(unused_tuple_struct_fields)] T);

fn main() {
    let x: Box<Node<[isize]>> = box Node([]);
}
