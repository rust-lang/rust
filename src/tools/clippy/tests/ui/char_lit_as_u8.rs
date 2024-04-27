#![warn(clippy::char_lit_as_u8)]

fn main() {
    // no suggestion, since a byte literal won't work.
    let _ = '❤' as u8;
    //~^ ERROR: casting a character literal to `u8` truncates
    //~| NOTE: `char` is four bytes wide, but `u8` is a single byte
}
