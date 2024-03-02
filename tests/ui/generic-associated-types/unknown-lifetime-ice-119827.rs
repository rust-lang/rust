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
//~| ERROR trait `Foo` is not implemented for `Box<dyn Foo>`
//~| ERROR not all trait items implemented

fn main() {}
