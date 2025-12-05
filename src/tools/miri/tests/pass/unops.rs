fn main() {
    assert_eq!(!true, false);
    assert_eq!(!0xFFu16, 0xFF00);
    assert_eq!(-{ 1i16 }, -1i16);
}
