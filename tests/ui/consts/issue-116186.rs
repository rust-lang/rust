#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

fn something(path: [usize; N]) -> impl Clone {
    //~^ ERROR cannot find value `N` in this scope
    match path {
        [] => 0,
        _ => 1,
    };
}

fn main() {}
