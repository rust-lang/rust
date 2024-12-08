// Regression test for
// https://github.com/rust-lang/rust/issues/67945#issuecomment-572617285

//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

struct Bug<S> { //[min]~ ERROR: parameter `S` is never used
    A: [(); { //[full]~ ERROR: overly complex generic constant
        let x: Option<Box<S>> = None;
        //[min]~^ ERROR: generic parameters may not be used in const operations
        0
    }],
}

fn main() {}
