//@ edition:2018
//@ revisions: full min

#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo;

impl<const H: feature> Foo {
//~^ ERROR: cannot find type `feature` in this scope
//~^^ ERROR: the const parameter `H` is not constrained by the impl trait, self type, or predicates
    async fn biz() {}
}

fn main() {}
