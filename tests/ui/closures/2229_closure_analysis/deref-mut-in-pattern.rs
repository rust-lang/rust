// Newly accepted examples as a result of the changes introduced in #138961.
//
//@ edition:2024
//@ check-pass
#![allow(unused_assignments)]

fn f() {
    let mut x: &mut [u8] = &mut [1, 2, 3];
    let c = || {
        match x {
            [] => (),
            _ => (),
        }
    };
    x = &mut [];
    c();
}

fn g() {
    let mut x: &mut bool = &mut false;
    let mut t = true;
    let c = || {
        match x {
            true => (),
            false => (),
        }
    };
    x = &mut t;
    c();
}

fn main() {
    f();
    g();
}
