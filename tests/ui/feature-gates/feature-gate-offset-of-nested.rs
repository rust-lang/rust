#![feature(offset_of_enum)]

use std::mem::offset_of;

struct S {
    a: u8,
    b: (u8, u8),
    c: T,
}

struct T {
    t: &'static str,
}

enum Alpha {
    One(u8),
    Two(u8),
}

fn main() {
    offset_of!(Alpha, Two.0); //~ ERROR only a single ident or integer is stable as the field in offset_of
    offset_of!(S, a);
    offset_of!((u8, S), 1);
    offset_of!((u32, (S, T)), 1.1); //~ ERROR only a single ident or integer is stable as the field in offset_of
    offset_of!(S, b.0); //~ ERROR only a single ident or integer is stable as the field in offset_of
    offset_of!((S, ()), 0.c); //~ ERROR only a single ident or integer is stable as the field in offset_of
    offset_of!(S, c.t); //~ ERROR only a single ident or integer is stable as the field in offset_of
}
