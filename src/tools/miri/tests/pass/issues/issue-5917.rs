struct T(&'static [isize]);
static STATIC: T = T(&[5, 4, 3]);
pub fn main() {
    let T(ref v) = STATIC;
    assert_eq!(v[0], 5);
}
