// run-pass

pub fn main() {
    let i: Box<_> = Box::new(vec![100]);
    assert_eq!((*i)[0], 100);
}
