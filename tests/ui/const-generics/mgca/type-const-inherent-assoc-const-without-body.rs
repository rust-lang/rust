//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

impl S { //~ ERROR cannot find type `S` in this scope
    #[type_const]
    const SIZE: usize;
    //~^ ERROR associated constant in `impl` without body
}

fn main() {}
