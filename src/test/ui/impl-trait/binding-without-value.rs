#![allow(incomplete_features)]
#![feature(impl_trait_in_bindings)]

fn foo() {
    let _ : impl Copy;
    //~^ ERROR cannot resolve opaque type
}

fn main() {}
