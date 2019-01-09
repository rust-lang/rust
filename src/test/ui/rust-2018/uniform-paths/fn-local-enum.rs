// compile-pass
// edition:2018

#![feature(uniform_paths)]

fn main() {
    enum E { A, B, C }

    use E::*;
    match A {
        A => {}
        B => {}
        C => {}
    }
}
