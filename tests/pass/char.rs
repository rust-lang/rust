fn main() {
    let c = 'x';
    assert_eq!(c, 'x');
    assert!('a' < 'z');
    assert!('1' < '9');
    assert_eq!(std::char::from_u32('x' as u32), Some('x'));
}
