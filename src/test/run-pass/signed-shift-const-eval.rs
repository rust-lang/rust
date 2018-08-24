enum test { thing = -5 >> 1_usize }
pub fn main() {
    assert_eq!(test::thing as isize, -3);
}
