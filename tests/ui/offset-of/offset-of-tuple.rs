//@ edition: 2021..
#![feature(forced_keywords, internal_syntax)]

use std::mem::offset_of;

fn main() {
    offset_of!((u8, u8), +1); //~ ERROR no rules expected
    offset_of!((u8, u8), -1); //~ ERROR offset_of expects dot-separated field and variant names
    offset_of!((u8, u8), 1.); //~ ERROR offset_of expects dot-separated field and variant names
    offset_of!((u8, u8), 1 .); //~ ERROR unexpected token: `)`
    // We need to put these into curly braces, otherwise only one of the
    // errors will be emitted and the others suppressed.
    { k#offset_of((u8, u8), +1) }; //~ ERROR leading `+` is not supported
    { k#offset_of((u8, u8), 1.) }; //~ ERROR offset_of expects dot-separated field and variant names
    { k#offset_of((u8, u8), 1 .) }; //~ ERROR unexpected token: `)`
}

type ComplexTup = (((u8, u8), u8), u8);

fn nested() {
    // All combinations of spaces (this sends different tokens to the parser)
    offset_of!(ComplexTup, 0.0.1.); //~ ERROR unexpected token: `)`
    offset_of!(ComplexTup, 0 .0.1.); //~ ERROR unexpected token: `)`
    offset_of!(ComplexTup, 0 . 0.1.); //~ ERROR unexpected token: `)`
    offset_of!(ComplexTup, 0. 0.1.); //~ ERROR unexpected token: `)`
    offset_of!(ComplexTup, 0.0 .1.); //~ ERROR unexpected token: `)`
    offset_of!(ComplexTup, 0.0 . 1.); //~ ERROR unexpected token: `)`
    offset_of!(ComplexTup, 0.0. 1.); //~ ERROR unexpected token: `)`

    // FIXME(fmease): Update comment.
    // Test for builtin too to ensure that the builtin syntax can also handle these cases
    // We need to put these into curly braces, otherwise only one of the
    // errors will be emitted and the others suppressed.
    { k#offset_of(ComplexTup, 0.0.1.) }; //~ ERROR unexpected token: `)`
    { k#offset_of(ComplexTup, 0 .0.1.) }; //~ ERROR unexpected token: `)`
    { k#offset_of(ComplexTup, 0 . 0.1.) }; //~ ERROR unexpected token: `)`
    { k#offset_of(ComplexTup, 0. 0.1.) }; //~ ERROR unexpected token: `)`
    { k#offset_of(ComplexTup, 0.0 .1.) }; //~ ERROR unexpected token: `)`
    { k#offset_of(ComplexTup, 0.0 . 1.) }; //~ ERROR unexpected token: `)`
    { k#offset_of(ComplexTup, 0.0. 1.) }; //~ ERROR unexpected token: `)`
}
