#![feature(inline_const_pat)]

// rust-lang/rust#82518: ICE with inline-const in match referencing const-generic parameter

fn foo<const V: usize>() {
    match 0 {
        const { V } => {},
        //~^ ERROR constant pattern depends on a generic parameter
        _ => {},
    }
}

const fn f(x: usize) -> usize {
    x + 1
}

fn bar<const V: usize>() {
    match 0 {
        const { f(V) } => {},
        //~^ ERROR constant pattern depends on a generic parameter
        _ => {},
    }
}

fn main() {
    foo::<1>();
    bar::<1>();
}
