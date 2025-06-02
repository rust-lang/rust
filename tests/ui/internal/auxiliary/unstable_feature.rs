#![allow(internal_features)]

#![feature(impl_stability)] 
#![feature(trivial_bounds)] // TODO: figure out what is this


pub trait Foo {
    fn foo();
}

pub struct Bar;

#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}

fn main() {}
