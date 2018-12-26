// compile-flags: --edition 2018

#![feature(try_blocks)]

fn main() {
    match try { false } { _ => {} } //~ ERROR expected expression, found reserved keyword `try`
}
