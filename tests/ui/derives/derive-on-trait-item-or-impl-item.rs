trait Foo {
    #[derive(Clone)]
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    type Bar;
}

struct Bar;

impl Bar {
    #[derive(Clone)]
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    fn bar(&self) {}
}

fn main() {}
