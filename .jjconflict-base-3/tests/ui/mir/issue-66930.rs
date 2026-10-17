//@ check-pass
//@ compile-flags: --emit=mir,link
// Regression test for #66930, this ICE requires `--emit=mir` flag.

static UTF8_CHAR_WIDTH: [u8; 0] = [];

pub fn utf8_char_width(b: u8) -> usize {
    UTF8_CHAR_WIDTH[b as usize] as usize
}

fn main() {}
