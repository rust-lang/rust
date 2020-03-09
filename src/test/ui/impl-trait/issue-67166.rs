// Regression test for #67166

#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

pub fn run() {
    let _foo: Box<impl Copy + '_> = Box::new(()); // FIXME: The error doesn't much make sense.
    //~^ ERROR: opaque type expands to a recursive type
}

fn main() {}
