//@ revisions: feature gated

#![cfg_attr(feature, feature(checked_type_aliases))]
#![allow(incomplete_features)]

type X = Vec<X>;
//[gated]~^ ERROR cycle detected
//[feature]~^^ ERROR: overflow normalizing the type alias `X`

#[rustfmt::skip]
fn main() { let b: X = Vec::new(); }
