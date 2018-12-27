// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#[derive(Clone)]
enum E {
    A,
    B(()),
    C
}

pub fn main() {
    let _ = E::A.clone();
}
