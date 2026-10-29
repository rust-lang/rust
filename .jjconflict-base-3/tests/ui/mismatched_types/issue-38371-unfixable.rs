fn ugh(&[bar]: &u32) {} //~ ERROR expected an array or slice

fn bgh(&&bar: u32) {} //~ ERROR mismatched types

fn main() {}
