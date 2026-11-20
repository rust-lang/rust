// Issue #141436
//@ run-rustfix
#![allow(dead_code)]

trait Trait<'a> {}

fn foo<T: Trait<'static>() {}
//~^ ERROR expected one of

fn main() {}
