use std::rc::Rc;

fn main() {
    let x = Rc::new([1, 2, 3, 4]);
    assert_eq!(*x, [1, 2, 3, 4]);
}
