// Warn when we encounter a manual `Default` impl that could be derived.
//@ run-rustfix
#![allow(dead_code)]
#![deny(default_could_be_derived)]
#![feature(default_field_values)]

// #[derive(Debug)]
// struct A;
// 
// impl Default for A {
//     fn default() -> Self { A }
// }
// 
#[derive(Debug)]
struct B(Option<i32>);

impl Default for B { //~ ERROR
    fn default() -> Self { B(Default::default()) }
}

#[derive(Debug)]
struct C(Option<i32>);

impl Default for C { //~ ERROR
    fn default() -> Self { C(None) }
}


// Explicit check against numeric literals and `Default::default()` calls.
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

// Explicit check against `None` literal, in the same way that we check against numeric literals.
#[derive(Debug)]
struct E {
    x: Option<i32>,
}

impl Default for E { //~ ERROR
    fn default() -> Self {
        E {
            x: None,
        }
    }
}

// Detection of unit variant ctors that could have been marked `#[default]`.
enum F<T> {
    Unit,
    Tuple(T),
}

impl<T> Default for F<T> { //~ ERROR
    fn default() -> Self {
        F::Unit
    }
}

// Comparison of `impl` *fields* with their `Default::default()` bodies.
struct G {
    f: F<i32>,
}

impl Default for G { //~ ERROR
    fn default() -> Self {
        G {
            f: F::Unit,
        }
    }
}

// Always lint against manual `Default` impl if all fields are defaulted.
#[derive(PartialEq, Debug)]
struct H {
    x: i32 = 101,
}

impl Default for H { //~ ERROR
    fn default() -> Self {
        H {
            x: 1,
        }
    }
}

// Always lint against manual `Default` impl if all fields are defaulted.
#[derive(PartialEq, Debug)]
struct I {
    x: i32 = 101,
    y: Option<i32>,
}

impl Default for I { //~ ERROR
    fn default() -> Self {
        I {
            x: 1,
            y: None,
        }
    }
}

fn main() {
//    let _ = A::default();
    let _ = B::default();
    let _ = C::default();
    let _ = D::default();
    let _ = E::default();
    let _ = F::<i32>::default();
    let _ = G::default();
    assert_eq!(H::default(), H { .. });
}
