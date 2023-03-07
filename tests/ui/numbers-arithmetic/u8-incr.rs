// run-pass




pub fn main() {
    let mut x: u8 = 12;
    let y: u8 = 12;
    x = x + 1;
    x = x - 1;
    assert_eq!(x, y);
    // x = 14;
    // x = x + 1;

}
