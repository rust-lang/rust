use super::*;
use std::cmp::Ordering;

#[derive(Debug)]
struct S(u32);

impl PartialEq for S {
    fn eq(&self, _other: &Self) -> bool {
        panic!("shouldn't be called");
    }
}

impl Eq for S {}

impl PartialOrd for S {
    fn partial_cmp(&self, other: &S) -> Option<Ordering> {
        // The `==` case should be handled by `Interned`.
        assert_ne!(self.0, other.0);
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for S {
    fn cmp(&self, other: &S) -> Ordering {
        // The `==` case should be handled by `Interned`.
        assert_ne!(self.0, other.0);
        self.0.cmp(&other.0)
    }
}

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

    assert_eq!(v1.cmp(&v2), Ordering::Less);
    assert_eq!(v3a.cmp(&v2), Ordering::Greater);
    assert_eq!(v1.cmp(&v1), Ordering::Equal); // only uses Interned::eq, not S::cmp
    assert_eq!(v3a.cmp(&v3b), Ordering::Equal); // only uses Interned::eq, not S::cmp

    assert_eq!(v1.partial_cmp(&v2), Some(Ordering::Less));
    assert_eq!(v3a.partial_cmp(&v2), Some(Ordering::Greater));
    assert_eq!(v1.partial_cmp(&v1), Some(Ordering::Equal)); // only uses Interned::eq, not S::cmp
    assert_eq!(v3a.partial_cmp(&v3b), Some(Ordering::Equal)); // only uses Interned::eq, not S::cmp
}
