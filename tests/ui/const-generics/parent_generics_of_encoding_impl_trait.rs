//@ aux-build:generics_of_parent_impl_trait.rs
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

extern crate generics_of_parent_impl_trait;

fn main() {
    // check for `impl Trait<{ const }>` which has a parent of a `DefKind::TyParam`
    generics_of_parent_impl_trait::foo([()]);
    //~^ error: type annotations needed
}
