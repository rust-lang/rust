use super::*;

// Are ASTs encodable?
#[test]
fn check_asts_encodable() {
    fn assert_encodable<
        T: for<'a> rustc_serialize::Encodable<rustc_serialize::json::Encoder<'a>>,
    >() {
    }
    assert_encodable::<Crate>();
}
