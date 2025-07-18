use std::path::PathBuf;

use crate::utils::cache::{INTERNER, Internable, Interner, TyIntern};

#[test]
fn test_string_interning() {
    let interner = Interner::default();
    let s1 = interner.intern_str("Hello");
    let s2 = interner.intern_str("Hello");
    let s3 = interner.intern_str("world");

    assert_eq!(s1, s2, "Same strings should be interned to the same instance");
    assert_ne!(s1, s3, "Different strings should have different interned values");
}

#[test]
fn test_interned_equality() {
    // Because we compare with &str, and the Deref impl accesses the global
    // INTERNER variable, we cannot use a local Interner variable here.
    let s1 = INTERNER.intern_str("test");
    let s2 = INTERNER.intern_str("test");

    assert_eq!(s1, s2);
    assert_eq!(s1, "test");
}

#[test]
fn test_ty_intern_intern_borrow() {
    let mut interner = TyIntern::default();
    let s1 = interner.intern_borrow("borrowed");
    let s2 = interner.intern("borrowed".to_string());

    assert_eq!(s1, s2);
    assert_eq!(interner.get(s1), "borrowed");
}
