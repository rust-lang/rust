//@ check-pass
//@compile-flags: -Clink-dead-code=true
// link-dead-code tries to eagerly monomorphize and collect items
// This lead to collector.rs to try and evaluate a type_const
// which will fail since they do not have bodies.
#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

const TYPE_CONST: usize = gca!(0);
fn main() {}
