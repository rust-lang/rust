//@ known-bug: #131534
#![feature(generic_const_exprs)]
type Value<'v> = &[[u8; SIZE]];

trait Trait: Fn(Value) -> Value {}
