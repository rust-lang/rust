// edition:2021

#![feature(async_fn_in_trait)]

trait MyTrait<'a, 'b, T> {
    async fn foo(&'a self, key: &'b T) -> (&'a ConnImpl, &'b T);
    //~^ ERROR: cannot find type `ConnImpl` in this scope [E0412]
}

impl<'a, 'b, T, U> MyTrait<T> for U {
    //~^ ERROR: implicit elided lifetime not allowed here [E0726]
    async fn foo(_: T) -> (&'a U, &'b T) {}
    //~^ ERROR: method `foo` has a `&self` declaration in the trait, but not in the impl [E0186]
    //~| ERROR: mismatched types [E0308]
}

fn main() {}
