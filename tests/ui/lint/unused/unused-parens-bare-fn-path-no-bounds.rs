//@ edition: 2018
//@ run-rustfix

#![deny(unused_parens)]
#![allow(warnings)]

type MultiBound = Box<Fn() -> (dyn Send + 'static)>;

type Nested = Box<dyn Fn() -> Fn() -> (dyn Send) + Sync>;

//~v ERROR unnecessary parentheses around type
type SingleBound = Box<Fn() -> (dyn Send)>;

fn main() {}
