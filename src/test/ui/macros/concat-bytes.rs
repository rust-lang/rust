// run-pass
#![feature(concat_bytes)]

fn main() {
    assert_eq!(concat_bytes!(), &[]);
    assert_eq!(concat_bytes!(b'A', b"BC", [68, b'E', 70]), b"ABCDEF");
}
