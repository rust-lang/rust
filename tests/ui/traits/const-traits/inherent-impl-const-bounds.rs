//@ check-pass
#![feature(const_trait_impl)]

struct S;

const trait A {}
const trait B {}

impl const A for S {}
impl const B for S {}

impl S {
    const fn a<T: [const] A>() where T: [const] B {

    }
}

const _: () = S::a::<S>();

fn main() {}
