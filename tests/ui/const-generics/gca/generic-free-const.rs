//@ check-pass
//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_items)]
#![expect(incomplete_features)]

const ADD1<const N: usize>: usize = N + 1;

fn a<const N: usize>() -> [usize; ADD1::<N>] {
    [ADD1::<N>; ADD1::<N>]
}

fn main() {
    let _: [(); ADD1::<1>] = [(); ADD1::<1>];
    let _: [(); ADD1::<{ ADD1::<1> }>] = [(); ADD1::<2>];
    a::<2>();
}
