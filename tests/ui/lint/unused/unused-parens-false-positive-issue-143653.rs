//@ check-pass

#![deny(unused_parens)]
#![allow(warnings)]
trait MyTrait {}

fn foo(_: Box<dyn FnMut(&mut u32) -> &mut (dyn MyTrait) + Send + Sync>) {}

fn main() {}
