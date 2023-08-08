#![feature(offset_of)]
#![feature(builtin_syntax)]

use std::mem::offset_of;

fn main() {
    offset_of!((u8, u8), _0); //~ ERROR no field `_0`
    offset_of!((u8, u8), 01); //~ ERROR no field `01`
    offset_of!((u8, u8), 1e2); //~ ERROR no field `1e2`
    offset_of!((u8, u8), 1_u8); //~ ERROR no field `1_`
    //~| ERROR suffixes on a tuple index
    offset_of!((u8, u8), +1); //~ ERROR no rules expected
    offset_of!((u8, u8), -1); //~ ERROR no rules expected
    offset_of!((u8, u8), 1.); //~ ERROR expected identifier, found `)`
    offset_of!((u8, u8), 1 .); //~ ERROR unexpected end of macro
    builtin # offset_of((u8, u8), 1e2); //~ ERROR no field `1e2`
    builtin # offset_of((u8, u8), _0); //~ ERROR no field `_0`
    builtin # offset_of((u8, u8), 01); //~ ERROR no field `01`
    builtin # offset_of((u8, u8), 1_u8); //~ ERROR no field `1_`
    //~| ERROR suffixes on a tuple index
    // We need to put these into curly braces, otherwise only one of the
    // errors will be emitted and the others suppressed.
    { builtin # offset_of((u8, u8), +1) }; //~ ERROR expected identifier, found `+`
    { builtin # offset_of((u8, u8), 1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of((u8, u8), 1 .) }; //~ ERROR expected identifier, found `)`
}

type ComplexTup = ((u8, (u8, u8)), u8);

fn nested() {
    offset_of!(((u8, u16), (u32, u16, u8)), 0.2); //~ ERROR no field `2`
    offset_of!(((u8, u16), (u32, u16, u8)), 1.2);
    offset_of!(((u8, u16), (u32, u16, u8)), 1.2.0); //~ ERROR no field `0`

    // All combinations of spaces (this sends different tokens to the parser)
    offset_of!(ComplexTup, 0.0.1.); //~ ERROR expected identifier
    offset_of!(ComplexTup, 0 .0.1.); //~ ERROR unexpected end of macro
    offset_of!(ComplexTup, 0 . 0.1.); //~ ERROR unexpected end of macro
    offset_of!(ComplexTup, 0. 0.1.); //~ ERROR no rules expected
    offset_of!(ComplexTup, 0.0 .1.); //~ ERROR expected identifier, found `)`
    offset_of!(ComplexTup, 0.0 . 1.); //~ ERROR expected identifier, found `)`
    offset_of!(ComplexTup, 0.0. 1.); //~ ERROR expected identifier, found `)`

    // Test for builtin too to ensure that the builtin syntax can also handle these cases
    // We need to put these into curly braces, otherwise only one of the
    // errors will be emitted and the others suppressed.
    { builtin # offset_of(ComplexTup, 0.0.1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of(ComplexTup, 0 .0.1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of(ComplexTup, 0 . 0.1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of(ComplexTup, 0. 0.1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of(ComplexTup, 0.0 .1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of(ComplexTup, 0.0 . 1.) }; //~ ERROR expected identifier, found `)`
    { builtin # offset_of(ComplexTup, 0.0. 1.) }; //~ ERROR expected identifier, found `)`
}
