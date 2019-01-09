// run-pass
#![allow(dead_code)]

enum E { V0, V1(isize) }
static C: &'static E = &E::V0;

pub fn main() {
    match *C {
        E::V0 => (),
        _ => panic!()
    }
}
