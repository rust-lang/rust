// run-pass

#![feature(c_str_literals)]

fn main() {
    assert_eq!(b"test\0", c"test".to_bytes_with_nul());
}
