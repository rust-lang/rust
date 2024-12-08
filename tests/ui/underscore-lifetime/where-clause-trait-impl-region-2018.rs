//@ run-rustfix
//@ edition:2018

#![allow(dead_code)]

trait WithType<T> {}
trait WithRegion<'a> { }

trait Foo { }

impl<T> Foo for Vec<T>
where
    T: WithType<&u32>
//~^ ERROR `&` without an explicit lifetime name cannot be used here
{ }

fn main() {}
