// run-pass

pub fn main() {
    let i: Box<_> = Box::new(100);
    let j = i;
    assert_eq!(*j, 100);
}
