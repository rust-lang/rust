#![feature(box_syntax)]

struct Node_ {
    a: Box<Cycle>
}

enum Cycle {
    Node(Node_),
    Empty,
}
fn main() {
    let mut x: Box<_> = box Cycle::Node(Node_ {a: box Cycle::Empty});
    // Create a cycle!
    match *x {
      Cycle::Node(ref mut y) => {
        y.a = x; //~ ERROR cannot move out of
      }
      Cycle::Empty => {}
    };
}
