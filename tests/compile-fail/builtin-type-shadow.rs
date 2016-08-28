#![feature(plugin)]
#![plugin(clippy)]
#![deny(builtin_type_shadow)]

fn foo<u32>(a: u32) -> u32 { //~ERROR shadows the built-in type `u32`
    42  //~ERROR E0308
    // ^ rustc's type error
}

fn main() {
}
