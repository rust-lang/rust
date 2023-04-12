// run-pass

pub fn main() {
    let i: Box<_> = Box::new(100);
    assert_eq!(*i, 100);
}
