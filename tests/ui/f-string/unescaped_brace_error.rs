#![feature(f_strings)]

pub fn main() {
    let a = f"foo}"; //~ ERROR invalid unescaped brace in f-string
}
