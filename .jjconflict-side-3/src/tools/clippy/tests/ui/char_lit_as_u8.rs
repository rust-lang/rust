#![warn(clippy::char_lit_as_u8)]

fn main() {
    let _ = 'a' as u8;
    //~^ char_lit_as_u8
    let _ = '\n' as u8;
    //~^ char_lit_as_u8
    let _ = '\0' as u8;
    //~^ char_lit_as_u8
    let _ = '\x01' as u8;
    //~^ char_lit_as_u8
}
