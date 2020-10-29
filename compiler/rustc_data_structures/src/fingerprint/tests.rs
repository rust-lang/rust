use super::*;

#[test]
fn test_value_roundtrip() {
    let f0 = Fingerprint::from_value(0x00221133_44665577, 0x88AA99BB_CCEEDDFF);
    let v = f0.to_value();
    let f1 = Fingerprint::from_value(v.0, v.1);
    assert_eq!(f0, f1);
}

#[test]
fn test_value_u128_roundtrip() {
    let f0 = Fingerprint::from_value_u128(0x00221133_44665577_88AA99BB_CCEEDDFF);
    let v = f0.to_value_u128();
    let f1 = Fingerprint::from_value_u128(v);
    assert_eq!(f0, f1);
}

#[test]
fn test_combine_commutative_is_commutative() {
    let f0 = Fingerprint::from_value_u128(0x00221133_44665577_88AA99BB_CCEEDDFF);
    let f1 = Fingerprint::from_value_u128(0x00112233_44556677_8899AABB_CCDDEEFF);
    assert_eq!(f0.combine_commutative(f1), f1.combine_commutative(f0));
}
