pub fn main() {
    let x = Box::new(10);
    let y = x;
    assert_eq!(*y, 10);
}
