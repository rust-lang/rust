//@ run-pass

const S: &str = "foo";
pub const B: &[u8] = S.as_bytes();
pub const C: usize = B.len();
pub const D: bool = B.is_empty();
pub const E: bool = S.is_empty();
pub const F: usize = S.len();

pub fn foo() -> [u8; S.len()] {
    let mut buf = [0; S.len()];
    for (i, &c) in S.as_bytes().iter().enumerate() {
        buf[i] = c;
    }
    buf
}

fn main() {
    assert_eq!(&foo()[..], b"foo");
    assert_eq!(foo().len(), S.len());
    const LEN: usize = S.len();
    assert_eq!(LEN, S.len());
    assert_eq!(B, foo());
    assert_eq!(B, b"foo");
    assert_eq!(C, 3);
    assert_eq!(F, 3);
    assert!(!D);
    assert!(!E);
    const EMPTY: bool = "".is_empty();
    assert!(EMPTY);
}
