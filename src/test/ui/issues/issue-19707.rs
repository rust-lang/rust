#![allow(dead_code)]

type Foo = fn(&u8, &u8) -> &u8; //~ ERROR missing lifetime specifier

fn bar<F: Fn(&u8, &u8) -> &u8>(f: &F) {} //~ ERROR missing lifetime specifier

fn main() {}
