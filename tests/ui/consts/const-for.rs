//@ check-pass
#![feature(const_trait_impl,const_iter,const_for)]

const _: () = {
    for _ in 0..5 {}
};

fn main() {}
