use super::utf16_to_utf8;

#[test]
fn zero_size_read() {
    assert_eq!(utf16_to_utf8(&[], &mut []).unwrap(), 0);
}
