// regression test, used to ICE

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

const N: usize = 4;

fn main() {
    let x = [(); N];
    //~^ ERROR use of `const` in the type system not defined as `type const`
}
