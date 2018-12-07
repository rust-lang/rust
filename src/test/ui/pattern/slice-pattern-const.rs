//compile-pass

fn main() {
    let s = &[0x00; 4][..]; //Slice of any value
    const MAGIC_TEST: &[u8] = b"TEST"; //Const slice to pattern match with
    match s {
        MAGIC_TEST => (),
        [0x00, 0x00, 0x00, 0x00] => (),
        [84, 69, 83, 84] => (), // this should warn
        _ => (),
    }
    match s {
        [0x00, 0x00, 0x00, 0x00] => (),
        MAGIC_TEST => (),
        [84, 69, 83, 84] => (), // this should warn
        _ => (),
    }
    match s {
        [0x00, 0x00, 0x00, 0x00] => (),
        [84, 69, 83, 84] => (),
        MAGIC_TEST => (), // this should warn
        _ => (),
    }
    const FOO: [u8; 1] = [4];
    match [99] {
        [0x00] => (),
        [4] => (),
        FOO => (), // this should warn
        _ => (),
    }
    const BAR: &[u8; 1] = &[4];
    match &[99] {
        [0x00] => (),
        [4] => (),
        BAR => (), // this should warn
        b"a" => (),
        _ => (),
    }

    const BOO: &[u8; 0] = &[];
    match &[] {
        [] => (),
        BOO => (), // this should warn
        b"" => (),
        _ => (),
    }
}
