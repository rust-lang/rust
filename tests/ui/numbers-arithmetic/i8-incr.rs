// run-pass




pub fn main() {
    let mut x: i8 = -12;
    let y: i8 = -12;
    x = x + 1;
    x = x - 1;
    assert_eq!(x, y);
}
