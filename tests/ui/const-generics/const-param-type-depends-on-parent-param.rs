#![feature(adt_const_params)]

trait Trait<const N: usize> {
    fn foo<const M: [u8; N]>() {}
    //~^ ERROR the type of const parameters must not depend on other generic parameters
}

fn main() {}
