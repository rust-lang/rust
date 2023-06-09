// run-pass
#![feature(concat_bytes)]

fn main() {
    assert_eq!(concat_bytes!(), &[]);
    assert_eq!(
        concat_bytes!(b'A', b"BC", [68, b'E', 70], [b'G'; 1], [72; 2], [73u8; 3], [65; 0]),
        b"ABCDEFGHHIII",
    );
    assert_eq!(
        concat_bytes!(
            concat_bytes!(b"AB", b"CD"),
            concat_bytes!(b"EF", b"GH"),
        ),
        b"ABCDEFGH",
    );
}
