//@ revisions: pass fail
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// Test that you can't leak unstable impls through item bounds on associated types.

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
  //[fail]~^ ERROR: unstable feature `feat_foo` is used without being enabled.

}

fn main(){}
