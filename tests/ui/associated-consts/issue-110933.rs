//@ check-pass

#![feature(associated_const_equality)]

pub trait Trait {
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
