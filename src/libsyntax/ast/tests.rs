use super::*;

// Are ASTs encodable?
#[test]
fn check_asts_encodable() {
    fn assert_encodable<T: rustc_serialize::Encodable>() {}
    assert_encodable::<Crate>();
}
