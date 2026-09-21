//@ check-pass
//@compile-flags: -Clink-dead-code=true
// link-dead-code tries to eagerly monomorphize and collect items
// This lead to collector.rs to try and evaluate a type_const
// which will fail since they do not have bodies.
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

const TYPE_CONST: usize = core::direct_const_arg!(0);
fn main() {}
