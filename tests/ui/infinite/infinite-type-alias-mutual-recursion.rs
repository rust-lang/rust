//@ revisions: current_feature next_feature current_gated next_gated
//@[next_feature] compile-flags: -Znext-solver
//@[next_gated] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

#![cfg_attr(any(current_feature, next_feature), feature(lazy_type_alias))]
#![allow(incomplete_features)]

type X1 = X2;
//[current_gated]~^ ERROR cycle detected when expanding type alias `X1`
//[next_gated]~^^ ERROR cycle detected when expanding type alias `X1`
//[current_feature]~^^^ ERROR: overflow normalizing the type alias `X2`
//[next_feature]~^^^^ ERROR: type mismatch resolving `X2 normalizes-to _`
type X2 = X3;
//[current_feature]~^ ERROR: overflow normalizing the type alias `X3`
//[next_feature]~^^ ERROR: type mismatch resolving `X3 normalizes-to _`

type X3 = X1;
//[current_feature]~^ ERROR: overflow normalizing the type alias `X1`
//[next_feature]~^^ ERROR: type mismatch resolving `X1 normalizes-to _`

fn main() {}
