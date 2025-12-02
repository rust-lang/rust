// Tests to make sure that we reject polymorphic dyn trait.
#![allow(todo_macro_calls)]

#![feature(non_lifetime_binders)]

trait Test<T> {}

fn foo() -> &'static dyn for<T> Test<T> {
    //~^ ERROR late-bound type parameter not allowed on trait object types
    todo!()
}

fn main() {}
