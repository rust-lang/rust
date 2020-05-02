#![warn(clippy::char_lit_as_u8)]

fn main() {
    let _ = '❤' as u8; // no suggestion, since a byte literal won't work.
}
