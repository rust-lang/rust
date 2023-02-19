// Tests to make sure that we reject polymorphic dyn trait.

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Test<T> {}

fn foo() -> &'static dyn for<T> Test<T> {
    //~^ ERROR late-bound type parameter not allowed on trait object types
    todo!()
}

fn main() {}
