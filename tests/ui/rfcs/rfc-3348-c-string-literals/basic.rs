//@ run-pass
//@ edition: 2021

fn main() {
    assert_eq!(b"test\0", c"test".to_bytes_with_nul());
}
