// check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct Foo<const A: usize, const B: usize>;

impl<const A: usize> Foo<1, A> {} // ok

fn main() {}
