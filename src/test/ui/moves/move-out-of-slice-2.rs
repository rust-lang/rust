#![feature(unsized_locals)]
//~^ WARN the feature `unsized_locals` is incomplete

struct A;
#[derive(Clone, Copy)]
struct C;

fn main() {
    let a: Box<[A]> = Box::new([A]);
    match *a {
        //~^ ERROR cannot move out of type `[A]`, a non-copy slice
        [a @ ..] => {}
        _ => {}
    }
    let b: Box<[A]> = Box::new([A, A, A]);
    match *b {
        //~^ ERROR cannot move out of type `[A]`, a non-copy slice
        [_, _, b @ .., _] => {}
        _ => {}
    }

    // `[C]` isn't `Copy`, even if `C` is.
    let c: Box<[C]> = Box::new([C]);
    match *c {
        //~^ ERROR cannot move out of type `[C]`, a non-copy slice
        [c @ ..] => {}
        _ => {}
    }
    let d: Box<[C]> = Box::new([C, C, C]);
    match *d {
        //~^ ERROR cannot move out of type `[C]`, a non-copy slice
        [_, _, d @ .., _] => {}
        _ => {}
    }
}
