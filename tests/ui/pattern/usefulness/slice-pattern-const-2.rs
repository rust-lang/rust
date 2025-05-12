#![deny(unreachable_patterns)]

fn main() {
    let s = &[0x00; 4][..]; //Slice of any value
    const MAGIC_TEST: &[u32] = &[4, 5, 6, 7]; //Const slice to pattern match with
    match s {
        MAGIC_TEST => (),
        [0x00, 0x00, 0x00, 0x00] => (),
        [4, 5, 6, 7] => (), //~ ERROR unreachable pattern
        _ => (),
    }
    match s {
        [0x00, 0x00, 0x00, 0x00] => (),
        MAGIC_TEST => (),
        [4, 5, 6, 7] => (), //~ ERROR unreachable pattern
        _ => (),
    }
    match s {
        [0x00, 0x00, 0x00, 0x00] => (),
        [4, 5, 6, 7] => (),
        MAGIC_TEST => (), //~ ERROR unreachable pattern
        _ => (),
    }
    const FOO: [u32; 1] = [4];
    match [99] {
        [0x00] => (),
        [4] => (),
        FOO => (), //~ ERROR unreachable pattern
        _ => (),
    }
}
