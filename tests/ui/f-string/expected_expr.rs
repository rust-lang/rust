#![feature(f_strings)]

pub fn main() {
    let a = f"foo{}bar"; //~ ERROR expected expression in f-string placeholder
}
