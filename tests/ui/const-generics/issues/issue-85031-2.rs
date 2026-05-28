//@ check-pass
//@ known-bug: unknown

// This should not compile, as the compiler should not know
// `A - 0` is satisfied `?x - 0` if `?x` is inferred to `A`.
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub struct Ref<'a>(&'a i32);

impl<'a> Ref<'a> {
    pub fn foo<const A: usize>() -> [(); A - 0] {
        Self::foo()
    }
}

fn main() {}
