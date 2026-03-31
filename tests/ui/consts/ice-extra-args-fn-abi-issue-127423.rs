// #127423: ICE: assertion failed: sig.c_variadic || extra_args.is_empty()
// Calling a const fn with the wrong number of arguments where the function has
// an unresolved type should not trigger an ICE.
// issue: rust-lang/rust#127423

#![allow(dead_code)]

const fn add(a: &'self isize) -> usize {
    //~^ ERROR use of undeclared lifetime name `'self`
    //~| ERROR lifetimes cannot use keyword names
    Qux + y
    //~^ ERROR cannot find value `Qux` in this scope
    //~| ERROR cannot find value `y` in this scope
}

const ARR: [i32; add(1, 2)] = [5, 6, 7];

pub fn main() {}
