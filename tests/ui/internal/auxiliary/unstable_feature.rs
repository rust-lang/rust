#![feature(impl_stability)] 

pub trait Foo {
    fn foo();
}

pub struct Bar;

#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}
