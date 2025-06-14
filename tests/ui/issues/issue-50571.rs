//@ edition: 2015
//@ run-rustfix

#![allow(dead_code)]
trait Foo {
    fn foo([a, b]: [i32; 2]) {}
    //~^ ERROR: patterns aren't allowed in methods without bodies
}

fn main() {}
