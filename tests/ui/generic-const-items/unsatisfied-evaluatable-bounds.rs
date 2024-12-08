#![feature(generic_const_items, generic_const_exprs)]
#![allow(incomplete_features)]

// Ensure that we check if (makeshift) "evaluatable"-bounds on const items hold or not.

const POSITIVE<const N: usize>: usize = N
where
    [(); N - 1]:; //~ ERROR evaluation of `POSITIVE::<0>::{constant#0}` failed

fn main() {
    let _ = POSITIVE::<0>;
}
