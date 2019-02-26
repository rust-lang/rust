const LEFT: u32 = 0x10000b3u32.rotate_left(8);
const RIGHT: u32 = 0xb301u32.rotate_right(8);

fn ident<T>(ident: T) -> T {
    ident
}

fn main() {
    assert_eq!(LEFT, ident(0xb301));
    assert_eq!(RIGHT, ident(0x10000b3));
}
