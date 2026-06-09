//@ revisions: feature_old gated_old feature_new gated_new
//@ ignore-parallel-frontend query cycle
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [feature_new] compile-flags: -Znext-solver
//@ [gated_new] compile-flags: -Znext-solver

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
#![cfg_attr(any(feature_old, feature_new), feature(lazy_type_alias))]
#![allow(incomplete_features)]

type X1 = X2;
//[gated_old,gated_new]~^ ERROR cycle detected when expanding type alias `X1`
//[feature_old]~^^ ERROR: overflow normalizing the type alias `X2`
//[feature_new]~^^^ ERROR: type mismatch resolving `X3 normalizes-to _`
type X2 = X3;
//[feature_old]~^ ERROR: overflow normalizing the type alias `X3`
//[feature_new]~^^ ERROR: type mismatch resolving `X1 normalizes-to _`
type X3 = X1;
//[feature_old]~^ ERROR: overflow normalizing the type alias `X1`
//[feature_new]~^^ ERROR: type mismatch resolving `X2 normalizes-to _`

fn main() {}
