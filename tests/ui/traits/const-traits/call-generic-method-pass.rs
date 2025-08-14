//! Basic test for calling methods on generic type parameters in `const fn`.

//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl, const_cmp)]

struct S;

impl const PartialEq for S {
    fn eq(&self, _: &S) -> bool {
        true
    }
    fn ne(&self, other: &S) -> bool {
        !self.eq(other)
    }
}

const fn equals_self<T: [const] PartialEq>(t: &T) -> bool {
    *t == *t
}

pub const EQ: bool = equals_self(&S);

fn main() {}
