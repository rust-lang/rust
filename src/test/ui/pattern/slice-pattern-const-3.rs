#![deny(unreachable_patterns)]

fn main() {
    let s = &["0x00"; 4][..]; //Slice of any value
    const MAGIC_TEST: &[&str] = &["4", "5", "6", "7"]; //Const slice to pattern match with
    match s {
        MAGIC_TEST => (),
        ["0x00", "0x00", "0x00", "0x00"] => (),
        ["4", "5", "6", "7"] => (), // FIXME(oli-obk): this should warn, but currently does not
        _ => (),
    }
    match s {
        ["0x00", "0x00", "0x00", "0x00"] => (),
        MAGIC_TEST => (),
        ["4", "5", "6", "7"] => (), // FIXME(oli-obk): this should warn, but currently does not
        _ => (),
    }
    match s {
        ["0x00", "0x00", "0x00", "0x00"] => (),
        ["4", "5", "6", "7"] => (),
        MAGIC_TEST => (), // FIXME(oli-obk): this should warn, but currently does not
        _ => (),
    }
    const FOO: [&str; 1] = ["boo"];
    match ["baa"] {
        ["0x00"] => (),
        ["boo"] => (),
        FOO => (), //~ ERROR unreachable pattern
        _ => (),
    }
}
