#![feature(auto_traits)]
#![feature(negative_impls)]
#![allow(bare_trait_objects)]

auto trait Auto {}

fn main() {
    let _: Box<((Auto)) + Auto>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `((Auto))`
    let _: Box<(Auto + Auto) + Auto>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(Auto + Auto)`
    let _: Box<(Auto +) + Auto>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(Auto)`
    let _: Box<(dyn Auto) + Auto>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(dyn Auto)`
}
