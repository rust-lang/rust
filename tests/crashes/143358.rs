//@ known-bug: rust-lang/rust#143358
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
fn identity<const T: identity<{ identity::<{ identity::<{}> }>() }>>();
