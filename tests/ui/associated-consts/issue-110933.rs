//@ check-pass

#![feature(associated_const_equality, min_generic_const_args)]
#![allow(incomplete_features)]

pub trait Trait {
    #[type_const]
    const ASSOC: usize;
}

pub fn foo<
    T: Trait<
        ASSOC = {
                    let a = 10_usize;
                    let b: &'_ usize = &a;
                    *b
                },
    >,
>() {
}

fn main() {}
