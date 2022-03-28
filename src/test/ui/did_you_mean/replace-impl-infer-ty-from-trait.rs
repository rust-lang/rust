// run-rustfix
#![allow(unused)]

trait Foo<T>: Sized {
    fn bar(i: i32, t: T, s: &Self) {}
}

impl Foo<usize> for () {
    fn bar(i: _, t: _, s: _) {}
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
}

fn main() {}
