// Warn when we encounter a manual `Default` impl that could be derived.
//@ run-rustfix
#![allow(dead_code)]
#![deny(default_could_be_derived)]

// #[derive(Debug)]
// struct A;
// 
// impl Default for A {
//     fn default() -> Self { A }
// }
// 
// #[derive(Debug)]
// struct B(Option<i32>);
// 
// impl Default for B {
//     fn default() -> Self { B(Default::default()) }
// }
// 
// #[derive(Debug)]
// struct C(Option<i32>);
// 
// impl Default for C {
//     fn default() -> Self { C(None) }
// }
// 

struct D {
    x: Option<i32>,
    y: i32,
}

impl Default for D { //~ ERROR
    fn default() -> Self {
        D {
            x: Default::default(),
            y: 0,
        }
    }
}
// 
// #[derive(Debug)]
// struct E {
//     x: Option<i32>,
// }
// 
// impl Default for E {
//     fn default() -> Self {
//         E {
//             x: None,
//         }
//     }
// }

enum F {
    Unit,
    Tuple(i32),
}

impl Default for F { //~ ERROR
    fn default() -> Self {
        F::Unit
    }
}

fn main() {
//    let _ = A::default();
//    let _ = B::default();
//    let _ = C::default();
//    let _ = D::default();
//    let _ = E::default();
    let _ = F::default();
}
