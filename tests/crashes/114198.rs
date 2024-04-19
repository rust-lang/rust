//@ known-bug: #114198
//@ compile-flags: -Zprint-mono-items=eager

#![feature(lazy_type_alias)]

impl Trait for Struct {}
trait Trait {
    fn test(&self) {}
}

type Struct = dyn Trait + Send;

fn main() {}
