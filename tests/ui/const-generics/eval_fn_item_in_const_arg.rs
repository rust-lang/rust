#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete and may not be safe to use and/or cause compiler crashes
#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete and may not be safe to use and/or cause compiler crashes

struct Checked<const N: usize, const M: usize = { N + 1 }>;
//~^ ERROR evaluation of `Checked::{constant#0}` failed

fn main() {
    let _: Checked<main>;
    //~^ ERROR the constant `main` is not of type `usize`
}
