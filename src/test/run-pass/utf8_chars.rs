use std::str;

pub fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: Vec<char> = vec!['e', 'é', '€', '\u{10000}'];
    let s: String = chs.iter().cloned().collect();
    let schs: Vec<char> = s.chars().collect();

    assert_eq!(s.len(), 10);
    assert_eq!(s.chars().count(), 4);
    assert_eq!(schs.len(), 4);
    assert_eq!(schs.iter().cloned().collect::<String>(), s);

    assert!((str::from_utf8(s.as_bytes()).is_ok()));
    // invalid prefix
    assert!((!str::from_utf8(&[0x80]).is_ok()));
    // invalid 2 byte prefix
    assert!((!str::from_utf8(&[0xc0]).is_ok()));
    assert!((!str::from_utf8(&[0xc0, 0x10]).is_ok()));
    // invalid 3 byte prefix
    assert!((!str::from_utf8(&[0xe0]).is_ok()));
    assert!((!str::from_utf8(&[0xe0, 0x10]).is_ok()));
    assert!((!str::from_utf8(&[0xe0, 0xff, 0x10]).is_ok()));
    // invalid 4 byte prefix
    assert!((!str::from_utf8(&[0xf0]).is_ok()));
    assert!((!str::from_utf8(&[0xf0, 0x10]).is_ok()));
    assert!((!str::from_utf8(&[0xf0, 0xff, 0x10]).is_ok()));
    assert!((!str::from_utf8(&[0xf0, 0xff, 0xff, 0x10]).is_ok()));
}
