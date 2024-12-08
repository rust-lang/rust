//@ run-pass
#![allow(dead_code)]

enum E { V, VV(isize) }
static C: E = E::V;

impl E {
    pub fn method(&self) {
        match *self {
            E::V => {}
            E::VV(..) => panic!()
        }
    }
}

pub fn main() {
    C.method()
}
