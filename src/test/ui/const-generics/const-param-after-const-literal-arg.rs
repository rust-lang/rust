// check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo<const A: usize, const B: usize>;

impl<const A: usize> Foo<1, A> {} // ok

fn main() {}
