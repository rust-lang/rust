trait Foo {
    //~^ ERROR cycle detected
    type Context<'c>
    where
        Self: 'c;
}

impl Foo for Box<dyn Foo> {}

fn main() {}
