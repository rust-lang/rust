// run-pass
#![allow(unused_variables)]

struct Node<T: ?Sized>(#[allow(unused_tuple_struct_fields)] T);

fn main() {
    let x: Box<Node<[isize]>> = Box::new(Node([]));
}
