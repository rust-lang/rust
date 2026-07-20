use super::*;

#[allow(unused)]
#[derive(Debug)]
struct S(u32);

impl PartialEq for S {
    fn eq(&self, _other: &Self) -> bool {
        panic!("shouldn't be called");
    }
}

impl Eq for S {}

#[test]
fn test_uniq() {
    let s1 = S(1);
    let s2 = S(2);
    let s3 = S(3);
    let s4 = S(1); // violates uniqueness

    let v1 = Interned::new_unchecked(&s1);
    let v2 = Interned::new_unchecked(&s2);
    let v3a = Interned::new_unchecked(&s3);
    let v3b = Interned::new_unchecked(&s3);
    let v4 = Interned::new_unchecked(&s4); // violates uniqueness

    assert_ne!(v1, v2);
    assert_ne!(v2, v3a);
    assert_eq!(v1, v1);
    assert_eq!(v3a, v3b);
    assert_ne!(v1, v4); // same content but different addresses: not equal
}
