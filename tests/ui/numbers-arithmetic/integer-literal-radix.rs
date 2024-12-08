//@ run-pass

pub fn main() {
    let a = 0xBEEF_isize;
    let b = 0o755_isize;
    let c = 0b10101_isize;
    let d = -0xBEEF_isize;
    let e = -0o755_isize;
    let f = -0b10101_isize;

    assert_eq!(a, 48879);
    assert_eq!(b, 493);
    assert_eq!(c, 21);
    assert_eq!(d, -48879);
    assert_eq!(e, -493);
    assert_eq!(f, -21);


}
