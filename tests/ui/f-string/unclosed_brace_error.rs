#![feature(f_strings)]

pub fn main() {
    let a = (f"foo{"); //~ ERROR invalid unclosed brace in f-string
}
