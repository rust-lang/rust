// edition:2018
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo;

impl<const H: feature> Foo {
//~^ ERROR: expected type, found built-in attribute `feature`
//~^^ ERROR: the const parameter `H` is not constrained by the impl trait, self type, or predicates
    async fn biz() {}
}

fn main() {}
