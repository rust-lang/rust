//@ check-pass

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

pub trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize;
}

pub fn foo<
    T: Trait<
        ASSOC = const {
                    let a = 10_usize;
                    let b: &'_ usize = &a;
                    *b
                },
    >,
>() {
}

fn main() {}
