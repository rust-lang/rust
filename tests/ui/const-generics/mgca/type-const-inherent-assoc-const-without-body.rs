//@ needs-rustc-debug-assertions

#![feature(gca_min_const_items)]
#![expect(incomplete_features)]

impl S { //~ ERROR cannot find type `S` in this scope
    const SIZE: usize;
    //~^ ERROR associated constant in `impl` without body
}

fn main() {}
