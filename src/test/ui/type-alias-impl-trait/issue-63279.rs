// compile-flags: -Zsave-analysis

#![feature(type_alias_impl_trait)]

type Closure = impl FnOnce();
//~^ ERROR could not find defining uses

fn c() -> Closure {
    || -> Closure { || () }
    //~^ ERROR: mismatched types
    //~| ERROR: expected a `FnOnce<()>` closure, found `()`
}

fn main() {}
