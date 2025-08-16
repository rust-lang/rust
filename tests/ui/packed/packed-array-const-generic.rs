//@ check-pass
#![allow(dead_code)]

#[repr(C, packed)]
struct PascalString<const CAP: usize> {
    len: u8,
    buf: [u8; CAP],
}

fn bar<const CAP: usize>(s: &PascalString<CAP>) -> &str {
    // 目标：这行不应触发 E0793
    std::str::from_utf8(&s.buf[0..s.len as usize]).unwrap()
}

fn main() {
    let p = PascalString::<10> { len: 3, buf: *b"abc\0\0\0\0\0\0\0" };
    let s = bar(&p);
    assert_eq!(s, "abc");
}
