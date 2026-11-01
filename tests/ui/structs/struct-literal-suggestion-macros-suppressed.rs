//@ forbid-output: use struct literal syntax instead of calling
// https://github.com/rust-lang/rust/issues/164044
#![allow(dead_code)]

struct A { b: Vec<u8>, c: usize }

macro_rules! constructor {
    () => { A };
}

fn main() {
    let _ = constructor!()(vec![], 2);
    //~^ ERROR cannot find function, tuple struct or tuple variant
}
