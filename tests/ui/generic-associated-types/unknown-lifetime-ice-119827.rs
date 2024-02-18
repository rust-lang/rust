trait Foo {
    type Context<'c>
    where
        Self: 'c;
}

impl Foo for Box<dyn Foo> {}
//~^ ERROR `Foo` cannot be made into an object
//~| ERROR `Foo` cannot be made into an object
//~| ERROR cycle detected
//~| ERROR cycle detected
//~| ERROR cycle detected
//~| ERROR the trait bound `Box<(dyn Foo + 'static)>: Foo` is not satisfied
//~| ERROR not all trait items implemented

fn main() {}
