#![feature(type_alias_impl_trait)]

type Closure = impl FnOnce();

#[defines(Closure)]
fn c() -> Closure {
    //~^ ERROR: expected a `FnOnce<()>` closure, found `()`
    || -> Closure { || () }
    //~^ ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: expected a `FnOnce<()>` closure, found `()`
}

fn main() {}
