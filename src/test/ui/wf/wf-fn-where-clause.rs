// Test that we check where-clauses on fn items.


#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

fn foo<T,U>() where T: ExtraCopy<U> //~ ERROR E0277
{
}

fn bar() where Vec<dyn Copy>:, {}
//~^ ERROR E0277
//~| ERROR E0038

struct Vec<T> {
    t: T,
}

fn main() { }
