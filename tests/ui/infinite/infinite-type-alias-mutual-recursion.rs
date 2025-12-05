//@ revisions: feature gated

#![cfg_attr(feature, feature(lazy_type_alias))]
#![allow(incomplete_features)]

type X1 = X2;
//[gated]~^ ERROR cycle detected when expanding type alias `X1`
//[feature]~^^ ERROR: overflow normalizing the type alias `X2`
type X2 = X3;
//[feature]~^ ERROR: overflow normalizing the type alias `X3`
type X3 = X1;
//[feature]~^ ERROR: overflow normalizing the type alias `X1`

fn main() {}
