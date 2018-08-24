use std::mem::swap;

pub fn main() {
    let mut x = 3; let mut y = 7;
    swap(&mut x, &mut y);
    assert_eq!(x, 7);
    assert_eq!(y, 3);
}
