#![feature(plugin)]
#![plugin(clippy)]
#![deny(builtin_type_shadow)]

fn foo<u32>(a: u32) -> u32 {
    42
    // ^ rustc's type error
}

fn main() {
}
