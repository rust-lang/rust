//@ check-pass

#![deny(unreachable_patterns)]

const C0: &'static [u8] = b"\x00";

fn main() {
    let x: &[u8] = &[0];
    match x {
        &[] => {}
        &[1..=255] => {}
        C0 => {}
        &[_, _, ..] => {}
    }
}
