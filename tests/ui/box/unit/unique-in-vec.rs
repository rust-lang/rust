// run-pass

pub fn main() {
    let vect : Vec<Box<_>> = vec![Box::new(100)];
    assert_eq!(vect[0], Box::new(100));
}
