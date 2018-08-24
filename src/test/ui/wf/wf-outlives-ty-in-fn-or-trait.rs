// Test that an appearance of `T` in fn args or in a trait object must
// still meet the outlives bounds. Since this is a new requirement,
// this is currently only a warning, not a hard error.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Trait<T> { }

struct Foo<'a,T> {
    f: &'a fn(T),
    //~^ ERROR E0309
}

struct Bar<'a,T> {
    f: &'a Trait<T>,
    //~^ ERROR E0309
}

#[rustc_error]
fn main() { }

