#![feature(offset_of)]

use std::mem::offset_of;

fn main() {
    offset_of!(NotEnoughArguments); //~ ERROR unexpected end of macro invocation
    offset_of!(NotEnoughArgumentsWithAComma, ); //~ ERROR unexpected end of macro invocation
    offset_of!(Container, field, too many arguments); //~ ERROR no rules expected the token `too`
    offset_of!(S, f); // compiles fine
    offset_of!(S, f,); // also compiles fine
    offset_of!(S, f.); //~ ERROR unexpected end of macro invocation
    offset_of!(S, f.,); //~ ERROR expected identifier
    offset_of!(S, f..); //~ ERROR no rules expected the token
    offset_of!(S, f..,); //~ ERROR no rules expected the token
    offset_of!(Lt<'static>, bar); // issue #111657
    offset_of!(Lt<'_>, bar); // issue #111678
}

struct S { f: u8, }
struct Lt<'a> {
    bar: &'a (),
}
