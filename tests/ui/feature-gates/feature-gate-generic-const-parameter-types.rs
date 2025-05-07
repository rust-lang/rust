//@ [feature] check-pass
//@ revisions: normal feature

#![cfg_attr(feature, feature(adt_const_params, generic_const_parameter_types))]
#![cfg_attr(feature, expect(incomplete_features))]

struct MyADT<const LEN: usize, const ARRAY: [u8; LEN]>;
//[normal]~^ ERROR: the type of const parameters must not depend on other generic parameters
//[normal]~| ERROR: `[u8; LEN]` is forbidden as the type of a const generic parameter

fn main() {}
