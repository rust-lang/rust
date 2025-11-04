#[test]
pub fn version() {
    let (major, _minor, _update) = core::char::UNICODE_VERSION;
    assert!(major >= 10);
}
