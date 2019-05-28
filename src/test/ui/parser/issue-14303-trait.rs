// run-rustfix
#![allow(dead_code)]
trait Foo<'a, T, 'b> {} //~ ERROR incorrect parameter order

fn main() {}
