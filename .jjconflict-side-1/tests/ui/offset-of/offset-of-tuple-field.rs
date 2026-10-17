#![feature(builtin_syntax)]

use std::mem::offset_of;

fn main() {
    offset_of!((u8, u8), _0); //~ ERROR no field `_0`
    offset_of!((u8, u8), 01); //~ ERROR no field `01`
    offset_of!((u8, u8), 1e2); //~ ERROR no field `1e2`
    offset_of!((u8, u8), 1_u8); //~ ERROR no field `1_`
    //~| ERROR suffixes on a tuple index

    builtin # offset_of((u8, u8), 1e2); //~ ERROR no field `1e2`
    builtin # offset_of((u8, u8), _0); //~ ERROR no field `_0`
    builtin # offset_of((u8, u8), 01); //~ ERROR no field `01`
    builtin # offset_of((u8, u8), 1_u8); //~ ERROR no field `1_`
    //~| ERROR suffixes on a tuple index

    offset_of!(((u8, u16), (u32, u16, u8)), 0.2); //~ ERROR no field `2`
    offset_of!(((u8, u16), (u32, u16, u8)), 0.1e2); //~ ERROR no field `1e2`
    offset_of!(((u8, u16), (u32, u16, u8)), 1.2);
    offset_of!(((u8, u16), (u32, u16, u8)), 1.2.0); //~ ERROR no field `0`
}
