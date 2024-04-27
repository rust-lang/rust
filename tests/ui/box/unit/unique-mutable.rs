//@ run-pass

pub fn main() {
    let mut i: Box<_> = Box::new(0);
    *i = 1;
    assert_eq!(*i, 1);
}
