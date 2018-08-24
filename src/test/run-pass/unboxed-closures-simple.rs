use std::ops::FnMut;

pub fn main() {
    let mut f = |x: isize, y: isize| -> isize { x + y };
    let z = f(1, 2);
    assert_eq!(z, 3);
}
