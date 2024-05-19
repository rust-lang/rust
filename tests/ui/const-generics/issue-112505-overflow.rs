#![feature(transmute_generic_consts)]

fn overflow(v: [[[u32; 8888888]; 9999999]; 777777777]) -> [[[u32; 9999999]; 777777777]; 239] {
    unsafe { std::mem::transmute(v) } //~ ERROR cannot transmute between types of different sizes
}

fn main() { }
