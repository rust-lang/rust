//@ run-pass

struct X { x: isize }

pub fn main() {
    let x: Box<_> = Box::new(X {x: 1});
    let bar = x;
    assert_eq!(bar.x, 1);
}
