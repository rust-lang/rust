//@ edition:2021


trait MyTrait<'a, 'b, T> {
    async fn foo(&'a self, key: &'b T) -> (&'a ConnImpl, &'b T);
    //~^ ERROR: cannot find type `ConnImpl` in this scope [E0412]
}

impl<'a, 'b, T, U> MyTrait<T> for U {
    //~^ ERROR: implicit elided lifetime not allowed here [E0726]
    async fn foo(_: T) -> (&'a U, &'b T) {}
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
