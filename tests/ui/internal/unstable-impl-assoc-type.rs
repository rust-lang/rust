#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// FIXME: add a description here.

trait Bar {}

trait Trait {
    type Assoc: Bar;
}

trait Moo {
    type Assoc: Bar;
}

struct Foo;

#[unstable_feature_bound(feat_foo)]
impl Bar for Foo {}

impl Trait for Foo {
  type Assoc = Self;
  //~^ ERROR: cannot satisfy `unstable feature: `feat_foo``
}

// If the impl is annotated with #[unstable_feature_bound(..)],
// then it should pass.
#[unstable_feature_bound(feat_foo)]
impl Moo for Foo {
  type Assoc = Self;
}

fn main(){}
