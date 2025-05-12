#![deny(unreachable_patterns)]

fn main() {
    let s = &[0x00; 4][..]; //Slice of any value
    const MAGIC_TEST: &[u8] = b"TEST"; //Const slice to pattern match with
    match s {
        MAGIC_TEST => (),
        [0x00, 0x00, 0x00, 0x00] => (),
        [84, 69, 83, 84] => (), //~ ERROR unreachable pattern
        _ => (),
    }
    match s {
        [0x00, 0x00, 0x00, 0x00] => (),
        MAGIC_TEST => (),
        [84, 69, 83, 84] => (), //~ ERROR unreachable pattern
        _ => (),
    }
    match s {
        [0x00, 0x00, 0x00, 0x00] => (),
        [84, 69, 83, 84] => (),
        MAGIC_TEST => (), //~ ERROR unreachable pattern
        _ => (),
    }
    const FOO: [u8; 1] = [4];
    match [99] {
        [0x00] => (),
        [4] => (),
        FOO => (), //~ ERROR unreachable pattern
        _ => (),
    }
    const BAR: &[u8; 1] = &[4];
    match &[99] {
        [0x00] => (),
        [4] => (),
        BAR => (), //~ ERROR unreachable pattern
        b"a" => (),
        _ => (),
    }

    const BOO: &[u8; 0] = &[];
    match &[] {
        [] => (),
        BOO => (), //~ ERROR unreachable pattern
        b"" => (), //~ ERROR unreachable pattern
        _ => (), //~ ERROR unreachable pattern
    }

    const CONST1: &[bool; 1] = &[true];
    match &[false] {
        CONST1 => {}
        [true] => {} //~ ERROR unreachable pattern
        [false] => {}
    }
}
