enum Enum {
    A,
    B,
}

impl const PartialEq for Enum {
    fn eq(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
}

#[test]
const fn const_discriminant_partial_eq() {
    assert!(Enum::A != Enum::B);
    assert!(Enum::A == Enum::A);
}
