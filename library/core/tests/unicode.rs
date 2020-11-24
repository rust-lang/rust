#[test]
pub fn version() {
    let (major, _minor, _update) = core::unicode::UNICODE_VERSION;
    assert!(major >= 10);
}
