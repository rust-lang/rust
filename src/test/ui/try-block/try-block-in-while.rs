// compile-flags: --edition 2018

#![feature(try_blocks)]

fn main() {
    while try { false } {} //~ ERROR expected expression, found reserved keyword `try`
}
