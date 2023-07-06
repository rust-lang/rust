// FIXME(c_str_literals): This should be `run-pass`
// known-bug: #113333
// edition: 2021

#![feature(c_str_literals)]

fn main() {
    assert_eq!(
        c"\xEF\x80🦀\u{1F980}".to_bytes_with_nul(),
        &[0xEF, 0x80, 0xF0, 0x9F, 0xA6, 0x80, 0xF0, 0x9F, 0xA6, 0x80, 0x00],
    );
}
