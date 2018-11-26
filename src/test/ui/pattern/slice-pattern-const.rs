// compile-pass

fn main() {
    let s = &[0x00; 4][..]; //Slice of any value
    const MAGIC_TEST: &[u8] = b"TEST"; //Const slice to pattern match with
    match s {
        MAGIC_TEST => (),
        [0x00, 0x00, 0x00, 0x00] => (),
        _ => (),
    }
}
