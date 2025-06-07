//@ revisions: pass fail
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// FIXME: add a description here.

trait Bar {}

trait Trait {
    type Assoc: Bar;
}

struct Foo;

#[unstable_feature_bound(feat_foo)]
impl Bar for Foo {}

#[cfg_attr(pass, unstable_feature_bound(feat_foo))]
impl Trait for Foo {
  type Assoc = Self;
  //[fail]~^ ERROR: cannot satisfy `unstable feature: `feat_foo``
}

fn main(){}
