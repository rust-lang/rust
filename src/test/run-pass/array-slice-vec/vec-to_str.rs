// run-pass


pub fn main() {
    assert_eq!(format!("{:?}", vec![0, 1]), "[0, 1]".to_string());

    let foo = vec![3, 4];
    let bar: &[isize] = &[4, 5];

    assert_eq!(format!("{:?}", foo), "[3, 4]");
    assert_eq!(format!("{:?}", bar), "[4, 5]");
}
