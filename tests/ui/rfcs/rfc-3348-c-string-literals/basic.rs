// FIXME(c_str_literals): This should be `run-pass`
// known-bug: #113333
// edition: 2021

#![feature(c_str_literals)]

fn main() {
    assert_eq!(b"test\0", c"test".to_bytes_with_nul());
}
