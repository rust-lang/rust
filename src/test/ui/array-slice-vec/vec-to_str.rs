// run-pass


pub fn main() {
    let bar: &[isize] = &[4, 5];
    assert_eq!(format!("{:?}", bar), "[4, 5]");
}
