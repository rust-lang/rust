//@ run-pass

struct Pair { x: isize, y: isize }

pub fn main() {
    for elt in &(vec![Pair {x: 10, y: 20}, Pair {x: 30, y: 0}]) {
        assert_eq!(elt.x + elt.y, 30);
    }
}
