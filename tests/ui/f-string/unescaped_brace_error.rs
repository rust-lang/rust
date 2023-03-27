#![feature(f_strings)]

pub fn main() {
    let a = f"foo}"; //~ ERROR invalid brace in f-string literal
}
