#![feature(box_syntax)]

struct node_ {
    a: Box<cycle>
}

enum cycle {
    node(node_),
    empty
}
fn main() {
    let mut x: Box<_> = box cycle::node(node_ {a: box cycle::empty});
    // Create a cycle!
    match *x {
      cycle::node(ref mut y) => {
        y.a = x; //~ ERROR cannot move out of
      }
      cycle::empty => {}
    };
}
