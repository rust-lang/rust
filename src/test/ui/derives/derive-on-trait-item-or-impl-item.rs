trait Foo {
    #[derive(Clone)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    type Bar;
}

struct Bar;

impl Bar {
    #[derive(Clone)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    fn bar(&self) {}
}

fn main() {}
