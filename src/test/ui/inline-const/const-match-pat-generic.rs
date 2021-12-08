#![allow(incomplete_features)]
#![feature(inline_const_pat)]
#![feature(generic_const_exprs)]

// rust-lang/rust#82518: ICE with inline-const in match referencing const-generic parameter

fn foo<const V: usize>() {
    match 0 {
        const { V } => {},
        //~^ ERROR const parameters cannot be referenced in patterns [E0158]
        _ => {},
    }
}

const fn f(x: usize) -> usize {
    x + 1
}

fn bar<const V: usize>() where [(); f(V)]: {
    match 0 {
        const { f(V) } => {},
        //~^ ERROR constant pattern depends on a generic parameter
        //~| ERROR constant pattern depends on a generic parameter
        _ => {},
    }
}

fn main() {
    foo::<1>();
    bar::<1>();
}
