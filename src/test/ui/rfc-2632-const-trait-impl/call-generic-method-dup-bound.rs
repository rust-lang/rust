// check-pass

#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

struct S;

impl const PartialEq for S {
    fn eq(&self, _: &S) -> bool {
        true
    }
    fn ne(&self, other: &S) -> bool {
        !self.eq(other)
    }
}

// This duplicate bound should not result in ambiguities. It should be equivalent to a single ~const
// bound.
const fn equals_self<T: PartialEq + ~const PartialEq>(t: &T) -> bool {
    *t == *t
}

trait A: PartialEq {}
impl<T: PartialEq> A for T {}

const fn equals_self2<T: A + ~const PartialEq>(t: &T) -> bool {
    *t == *t
}

pub const EQ: bool = equals_self(&S) && equals_self2(&S);

fn main() {}
