// Regression test for #67166

#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

pub fn run() {
    let _foo: Box<impl Copy + '_> = Box::new(());
    //~^ ERROR: missing lifetime specifier
}

fn main() {}
