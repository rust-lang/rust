//@ run-pass


pub fn main() {
    let (&x, &y) = (&3, &'a');
    assert_eq!(x, 3);
    assert_eq!(y, 'a');
}
