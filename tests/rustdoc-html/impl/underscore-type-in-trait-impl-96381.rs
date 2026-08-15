//@ should-fail
// https://github.com/rust-lang/rust/issues/96381

#![allow(unused)]

trait Foo<T>: Sized {
    fn bar(i: i32, t: T, s: &Self) -> (T, i32);
}

impl Foo<usize> for () {
    fn bar(i: _, t: _, s: _) -> _ {
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
        (1, 2)
    }
}

fn main() {}
