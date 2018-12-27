// compile-pass
// edition:2018

fn main() {
    enum E { A, B, C }

    use E::*;
    match A {
        A => {}
        B => {}
        C => {}
    }
}
