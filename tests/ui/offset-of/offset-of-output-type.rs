use std::mem::offset_of;

struct S {
    v: u8,
    w: u16,
}


fn main() {
    let _: u8 = offset_of!(S, v); //~ ERROR mismatched types
    let _: u16 = offset_of!(S, v); //~ ERROR mismatched types
    let _: u32 = offset_of!(S, v); //~ ERROR mismatched types
    let _: u64 = offset_of!(S, v); //~ ERROR mismatched types
    let _: isize = offset_of!(S, v); //~ ERROR mismatched types
    let _: usize = offset_of!(S, v);

    offset_of!(S, v) //~ ERROR mismatched types
}
