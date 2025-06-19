#![allow(unused)]

struct A;
#[derive(Clone, Copy)]
struct C;

fn main() {
    let a: Box<[A]> = Box::new([A]);
    match *a {
        [a @ ..] => {} //~ERROR the size for values of type `[A]` cannot be known at compilation time [E0277]
        _ => {}
    }
    let b: Box<[A]> = Box::new([A, A, A]);
    match *b {
        [_, _, b @ .., _] => {} //~ERROR the size for values of type `[A]` cannot be known at compilation time [E0277]
        _ => {}
    }

    // `[C]` isn't `Copy`, even if `C` is.
    let c: Box<[C]> = Box::new([C]);
    match *c {
        [c @ ..] => {} //~ERROR the size for values of type `[C]` cannot be known at compilation time [E0277]
        _ => {}
    }
    let d: Box<[C]> = Box::new([C, C, C]);
    match *d {
        [_, _, d @ .., _] => {} //~ERROR the size for values of type `[C]` cannot be known at compilation time [E0277]
        _ => {}
    }
}
