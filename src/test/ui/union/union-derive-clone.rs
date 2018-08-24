#![feature(untagged_unions)]

#[derive(Clone)] //~ ERROR the trait bound `U1: std::marker::Copy` is not satisfied
union U1 {
    a: u8,
}

#[derive(Clone)]
union U2 {
    a: u8, // OK
}

impl Copy for U2 {}

#[derive(Clone, Copy)]
union U3 {
    a: u8, // OK
}

#[derive(Clone, Copy)]
union U4<T> {
    a: T, // OK
}

#[derive(Clone)]
struct CloneNoCopy;

fn main() {
    let u = U4 { a: CloneNoCopy };
    let w = u.clone(); //~ ERROR no method named `clone` found for type `U4<CloneNoCopy>`
}
