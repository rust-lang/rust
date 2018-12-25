pub fn main() {
    let c: char = 'x';
    let d: char = 'x';
    assert_eq!(c, 'x');
    assert_eq!('x', c);
    assert_eq!(c, c);
    assert_eq!(c, d);
    assert_eq!(d, c);
    assert_eq!(d, 'x');
    assert_eq!('x', d);
}
