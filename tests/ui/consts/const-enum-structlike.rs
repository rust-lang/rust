// run-pass
#![allow(dead_code)]

enum E {
    S0 { s: String },
    S1 { u: usize }
}

static C: E = E::S1 { u: 23 };

pub fn main() {
    match C {
        E::S0 { .. } => panic!(),
        E::S1 { u } => assert_eq!(u, 23)
    }
}
