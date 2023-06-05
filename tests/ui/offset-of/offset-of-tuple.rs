#![feature(offset_of)]
#![feature(builtin_syntax)]

fn main() {
    core::mem::offset_of!((u8, u8), _0); //~ ERROR no field `_0`
    core::mem::offset_of!((u8, u8), +1); //~ ERROR no rules expected
    core::mem::offset_of!((u8, u8), -1); //~ ERROR no rules expected
    builtin # offset_of((u8, u8), _0); //~ ERROR no field `_0`
    builtin # offset_of((u8, u8), +1); //~ ERROR expected identifier
}
